import torch
import torch.nn as nn
from packaging import version
import numpy as np

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"


class IdentityWrapper(nn.Module):
    def __init__(self, diffusion_model, compile_model: bool = False):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0"))
            and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        if "cond_view" in c:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                cond_view=c.get("cond_view", None),
                cond_motion=c.get("cond_motion", None),
                **kwargs,
            )
        else:
            return self.diffusion_model(
                x,
                timesteps=t,
                context=c.get("crossattn", None),
                y=c.get("vector", None),
                **kwargs,
            )


class MONAIWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            class_labels=c.get("scalar", None),
            **kwargs,
        )


class GNNWrapper(IdentityWrapper):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        # batch_size = kwargs["batch_size"]
        in_shape = x.shape
        batch_size = x.shape[0]
        x = x.flatten(0, 1)
        num_nodes = 400
        new_computed_edge_indices = []
        for i in range(batch_size):
            computed_edges = self.compute_edges(
                x[i * (num_nodes) : (i + 1) * num_nodes]
            )
            computed_edges += i * num_nodes
            new_computed_edge_indices.append(computed_edges)

        concat_new_indices = torch.cat(new_computed_edge_indices, axis=1)
        edge_index = concat_new_indices

        out = self.diffusion_model(x=x, t=t, edge_index=edge_index, **kwargs)
        return out.view(in_shape)

    def compute_edges(self, corr, threshold=5):
        """construct adjacency matrix from the given correlation matrix and threshold. Taken from NeuroGraph code: https://github.com/Anwar-Said/NeuroGraph/blob/main/NeuroGraph/preprocess.py#L274
        Threshold is set to top 5%, for HCPTask dataset."""
        corr_matrix_copy = corr.clone()
        threshold = torch.quantile(
            corr_matrix_copy[corr_matrix_copy > 0], (100 - threshold) / 100.0
        )
        corr_matrix_copy[corr_matrix_copy < threshold] = 0
        corr_matrix_copy[corr_matrix_copy >= threshold] = 1
        return corr_matrix_copy.nonzero().t().to(torch.long)
