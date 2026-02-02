import torch
import torch.nn as nn
import os
from packaging import version
from ...util import instantiate_from_config

try:
    from data_utils.utils import compute_edges
except ModuleNotFoundError:
    print(
        "Warning: data_utils module not found. Can't run GNNWrapper. Copy the module from the parent repo, or implement the missing methods."
    )

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
    def __init__(
        self,
        diffusion_model,
        compile_model: bool = False,
        static_edge_path: str = None,
    ):
        if not isinstance(diffusion_model, nn.Module):
            diffusion_model = instantiate_from_config(diffusion_model)
        super().__init__(diffusion_model, compile_model=compile_model)
        self.static_edge_index = None
        if static_edge_path is not None:
            if os.path.exists(static_edge_path):
                self.static_edge_index = torch.load(static_edge_path)
                print(f"GNNWrapper: Loaded static edges from {static_edge_path}")
            else:
                print(
                    f"GNNWrapper: Warning - Static edge path {static_edge_path} not found. Will compute dynamically."
                )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs
    ) -> torch.Tensor:
        # Assumes data is in shape (B, N, C)
        in_shape = x.shape
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        x = x.flatten(0, 1)

        new_computed_edge_indices = []
        if self.static_edge_index is not None:
            if self.static_edge_index.device != x.device:
                self.static_edge_index = self.static_edge_index.to(x.device)

            for i in range(batch_size):
                # Offset the static edges for each graph in the batch
                new_computed_edge_indices.append(self.static_edge_index + i * num_nodes)
        else:
            for i in range(batch_size):
                computed_edges = compute_edges(x[i * num_nodes : (i + 1) * num_nodes])
                computed_edges += i * num_nodes
                new_computed_edge_indices.append(computed_edges)

        edge_index = torch.cat(new_computed_edge_indices, axis=1)

        if "batch" not in kwargs.keys():
            batch = torch.arange(batch_size, device=x.device).repeat_interleave(
                num_nodes
            )
        else:
            batch = kwargs.pop("batch")

        out = self.diffusion_model(
            x=x,
            t=t,
            edge_index=edge_index,
            context=c.get("vector"),
            batch=batch,
            **kwargs,
        )
        return out.view(in_shape)
