# transformers/models/llama/modeling_llama.py, set use_reentrant to False

import os
from pathlib import Path
from typing import List, TextIO, Tuple, cast
from uuid import uuid4

import torch
import torch.nn.functional as F
from accelerate import Accelerator  # type: ignore
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (  # type: ignore
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

import wandb
from cli.settings import BUILD_DIR
from dataset.models import LineProbability, TrainingData, TrainingDataList


class Analyzer:
    def __init__(
        self,
        pretrained_model_name_or_path: Path,
        devices: List[str],
        len_soft_prompt: int | None = None,
        soft_prompt_path: Path | None = None,
    ):
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._devices = devices
        self._len_soft_prompt = len_soft_prompt
        self._soft_prompt_path = soft_prompt_path

        self._name = pretrained_model_name_or_path.stem

        self._accelerator = Accelerator()

        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(  # type: ignore
                pretrained_model_name_or_path,
                trust_remote_code=True,
                use_fast=True,
            )
        )

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(  # type: ignore
            pretrained_model_name_or_path,
            device_map="auto",
            config={"pad_token_id": True},
        )

        self._model = self._accelerator.prepare(self._model)  # type: ignore

        self._embedding: torch.nn.Embedding = cast(torch.nn.Embedding, self._model.get_input_embeddings())  # type: ignore

        for parameter in self._model.parameters(True):  # type: ignore
            parameter.requires_grad_(False)  # type: ignore

        self._soft_prompt: torch.nn.Embedding | None = None

        if self._soft_prompt_path is not None:
            self._soft_prompt = cast(torch.nn.Embedding, torch.load(self._soft_prompt_path, map_location=self._devices[0]))  # type: ignore
            self._len_soft_prompt = self._soft_prompt.num_embeddings
        elif self._len_soft_prompt is not None:
            self._soft_prompt = torch.nn.Embedding(
                num_embeddings=self._len_soft_prompt,
                embedding_dim=self._embedding.embedding_dim,
                device=self._devices[0],
            )
            self._soft_prompt.weight = torch.nn.Parameter(
                self._embedding.weight[: self._len_soft_prompt].clone().detach()
            )

        self._tokenizer.pad_token = self._tokenizer.eos_token

    @property
    def name(self):
        return self._name

    def save_soft_prompt(self, path: Path):
        torch.save(self._soft_prompt, path)  # type: ignore

    def probability_of_line(
        self,
        batched_lines: List[TrainingData],
        until: int | None = None,
        requires_grad: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        all_embedded_tokens = self._embedding(
            torch.tensor(
                [_ for _ in range(self._model.config.vocab_size)],
                device=self._devices[0],
            )
        )

        previous_lines = [line.prefix for line in batched_lines]

        encoded_previous_lines = self._tokenizer(
            previous_lines, return_tensors="pt", padding=True, truncation=True
        )

        input_embeds_prefix = self._embedding(
            cast(torch.Tensor, encoded_previous_lines["input_ids"]).to(
                device=self._devices[0]
            )
        )
        attention_mask_prefix: torch.Tensor = cast(
            torch.Tensor, encoded_previous_lines["attention_mask"]
        ).to(device=self._devices[0])

        encoded_lines = self._tokenizer(
            [line.value for line in batched_lines],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        input_ids_lines: torch.Tensor = encoded_lines["input_ids"]  # type: ignore

        attention_mask_lines: torch.Tensor = encoded_lines["attention_mask"]  # type: ignore

        length_of_lines = input_ids_lines.shape[1]

        last_expected_vectors = None

        mutable_result = torch.zeros(len(batched_lines), device=self._devices[0])

        for col in tqdm(range(length_of_lines), desc="Column", leave=False):
            current_input_ids = input_ids_lines[:, [col]]
            current_attention_mask = attention_mask_lines[:, [col]]

            if last_expected_vectors is None:
                input_embeds = input_embeds_prefix
                attention_mask = attention_mask_prefix
            else:
                input_embeds = torch.cat(
                    [
                        input_embeds_prefix.to(device=self._devices[0]),
                        *(
                            [
                                self._soft_prompt.weight.repeat(
                                    last_expected_vectors.size(0), 1, 1
                                )
                            ]
                            if self._soft_prompt is not None
                            else []
                        ),
                        last_expected_vectors.to(device=self._devices[0]),
                    ],
                    dim=1,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask_prefix.to(device=self._devices[0]),
                        *(
                            [
                                torch.ones(
                                    input_embeds.size(0),
                                    self._soft_prompt.num_embeddings,
                                    device=self._devices[0],
                                )
                            ]
                            if self._soft_prompt is not None
                            else []
                        ),
                        current_attention_mask.to(device=self._devices[0]),
                    ],
                    dim=1,
                )

            if requires_grad:
                input_embeds = input_embeds.requires_grad_(True)
            else:
                input_embeds = input_embeds.detach()

            output = self._model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            )

            logits = output.logits[:, -1, :]

            token_log_probs = torch.log_softmax(logits, -1)
            token_probs = torch.softmax(logits, -1)

            last_expected_vectors = torch.matmul(
                token_probs.to(self._devices[0]),
                all_embedded_tokens.to(self._devices[0]),
            ).unsqueeze(1)

            diagonal_attention_mask = torch.diag(current_attention_mask.squeeze(1))

            vectorized_current_input_ids = F.one_hot(
                current_input_ids,
                self._model.config.vocab_size,
            ).squeeze(1)

            mask = (
                torch.matmul(diagonal_attention_mask, vectorized_current_input_ids)
                .to(torch.float)
                .to(device=self._devices[0])
            )

            current_result = (
                torch.matmul(mask, token_log_probs.transpose(0, 1))
                .diag()
                .to(device=self._devices[0])
            )

            if until is not None and col >= until:
                continue

            mutable_result += current_result

        return mutable_result, length_of_lines

    @staticmethod
    def criterion(
        batched_log_probs: torch.Tensor,
        is_vulnerable: List[bool],
    ):
        number_of_vulnerable = 279  # FIXME
        number_of_not_vulnerable = 2554  # FIXME

        batch_size = len(is_vulnerable)

        vulnerable_mask = torch.diag(
            torch.tensor(
                is_vulnerable, dtype=torch.bool, device=batched_log_probs.device
            )
        ).to(torch.float)

        not_vulnerable_mask = torch.diag(
            ~torch.tensor(
                is_vulnerable, dtype=torch.bool, device=batched_log_probs.device
            )
        ).to(torch.float)

        vulnerable_only = torch.matmul(
            batched_log_probs.to(torch.float),
            vulnerable_mask.to(torch.float),
        ).sum(-1)

        not_vulnerable_only = torch.matmul(
            -batched_log_probs.to(torch.float),
            not_vulnerable_mask.to(torch.float),
        ).sum(-1) * (number_of_vulnerable / number_of_not_vulnerable)

        return (vulnerable_only + not_vulnerable_only) / (
            batch_size * 100  # FIXME: Magic Number
        )

    def train(
        self,
        train_dataloader: DataLoader[TrainingData],
        test_dataloader: DataLoader[TrainingData],
        lr: float,
        num_epochs: int,
        batch_size: int,
        epoch_start: int = 0,
        accumulate_grad_batches: int = 1,
        max_norm: float = 5.0,
        until: int | None = None,
    ):
        attempt_key = f"{uuid4().hex[:4]}-{self._len_soft_prompt}-{lr}-{batch_size}-{accumulate_grad_batches}-{max_norm}-{until}"

        wandb.init(  # type: ignore
            project="CSED499",
            config={
                "learning_rate": lr,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "accumulate_grad_batches": accumulate_grad_batches,
                "max_norm": max_norm,
                "len_soft_prompt": self._len_soft_prompt,
                "until": until,
            },
            tags=[
                f"lr-{lr}",
                f"batch_size-{batch_size}",
                f"max_norm-{max_norm}",
                f"accumulate_grad_batches-{accumulate_grad_batches}",
                f"len_soft_prompt-{self._len_soft_prompt}",
            ],
            name=f"{self.name}-{attempt_key}",
        )

        self._model.gradient_checkpointing_enable()  # type: ignore

        optimizer = torch.optim.AdamW(self._model.parameters(True), lr=lr)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        optimizer, train_dataloader, lr_scheduler = cast(
            Tuple[
                torch.optim.Optimizer,
                DataLoader[TrainingData],
                torch.optim.lr_scheduler.LRScheduler,
            ],
            self._accelerator.prepare(optimizer, train_dataloader, lr_scheduler),  # type: ignore
        )

        if self._soft_prompt is not None:
            self._soft_prompt.weight.requires_grad_(True)
            self._soft_prompt.requires_grad_(True)

        for epoch in tqdm(range(epoch_start, epoch_start + num_epochs), desc="Epoch"):
            self._model.train()

            train_loss = torch.tensor(0.0, device=self._devices[0])

            last_train_loss = torch.tensor(0.0, device=self._devices[0])

            for step, batch in enumerate(
                tqdm(train_dataloader, desc=f"Train at Epoch {epoch}")
            ):
                batch = TrainingDataList.validate_python(batch)

                is_vulnerable = [line.is_vulnerable for line in batch]

                output, _ = self.probability_of_line(batch, until, requires_grad=True)

                loss = self.criterion(output, is_vulnerable)
                loss = loss / accumulate_grad_batches

                self._accelerator.backward(loss)  # type: ignore

                torch.nn.utils.clip_grad_norm_(self._model.parameters(True), max_norm)  # type: ignore

                train_loss += loss.clone().detach().float()

                if ((step + 1) % accumulate_grad_batches == 0) or (
                    (step + 1) == len(train_dataloader)
                ):
                    optimizer.step()
                    optimizer.zero_grad()

                    lr_scheduler.step(train_loss - last_train_loss)
                    self._model.zero_grad()

                    wandb.log(  # type: ignore
                        {
                            "train/loss": train_loss - last_train_loss,
                            "train/lr": lr_scheduler.get_last_lr()[0],
                        }
                    )

                    last_train_loss = train_loss.clone().detach()

                if step % 100 == 0 or (step + 1) == len(train_dataloader):
                    print(f"{step=}: {loss=}")
                    self._accelerator.wait_for_everyone()

                    output_dir = (
                        self._pretrained_model_name_or_path.parent
                        / f"{self.name}-{attempt_key}"
                    )

                    os.makedirs(output_dir, exist_ok=True)
                    self.save_soft_prompt(output_dir / f"{epoch}-{step}.pt")

            with torch.no_grad():
                self._model.eval()

                output_file = open(
                    BUILD_DIR
                    / f"line-probabilities-{self.name}-{attempt_key}-{epoch}.csv".lower(),
                    "w",
                )

                eval_loss = self.evaluate(test_dataloader, output_file)

                output_file.close()

                eval_epoch_loss = eval_loss / len(test_dataloader)

                eval_ppl = torch.exp(eval_epoch_loss)
                train_epoch_loss = train_loss / len(train_dataloader)
                train_ppl = torch.exp(train_epoch_loss)

                print(
                    f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}"
                )

    def evaluate(
        self,
        test_dataloader: DataLoader[TrainingData],
        output_file: TextIO,
    ):
        writer = LineProbability.use_csv_dict_writer_with_file(output_file)

        loss = torch.tensor(0.0, device=self._devices[0])

        for batch in tqdm(test_dataloader, desc="Eval"):
            batch = TrainingDataList.validate_python(batch)

            is_vulnerable = [line.is_vulnerable for line in batch]

            with torch.no_grad():
                output, _ = self.probability_of_line(batch)

                loss += self.criterion(output, is_vulnerable).clone().detach().float()

                probabilities = cast(List[float], output.tolist())  # type: ignore

                for data, probability in zip(batch, probabilities):
                    LineProbability(
                        line_number=-1,
                        probability=probability,
                        is_vulnerable=data.is_vulnerable,
                        model=self.name,
                        line=data.value,
                        prefix=data.prefix,
                        rule_id=data.rule_id,
                    ).write_csv_row(writer)

                    output_file.flush()

        return loss
