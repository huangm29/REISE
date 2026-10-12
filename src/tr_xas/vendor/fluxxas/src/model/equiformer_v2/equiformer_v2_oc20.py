import logging
import math
from typing import Dict, List, Optional
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from .utils import compute_neighbors, get_pbc_distances, radius_graph_pbc

from .edge_rot_mat import init_edge_rot_mat
from .gaussian_rbf import GaussianRadialBasisLayer
from .input_block import EdgeDegreeEmbedding
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .module_list import ModuleListInfo
from .radial_function import RadialFunction
from .smearing import GaussianSmearing
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_LinearV2,
    SO3_Rotation,
)
from .transformer_block import (
    FeedForwardNetwork,
    SO2EquivariantGraphAttention,
    TransBlockV2,
)

# Statistics of nablaDFT train 100K
_AVG_NUM_NODES = 39.65745326960467
_AVG_DEGREE = 19.16009564536883


class EquiformerV2_OC20(nn.Module):
    """Equiformer with graph attention built upon SO(2) convolution and
    feedforward network built upon S2 activation.

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number
        xas_feature_dim (int):  Dimension of XAS spectral features

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int): Number of samples used to approximate the integration of
            the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative
            distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel
            share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise,
            use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear
            layers except those in radial functions
        enforce_max_neighbors_strictly (bool):      When edges are subselected based on the `max_neighbors` arg,
            arbitrarily select amongst equidistant / degenerate edges to have exactly the correct number.
        avg_num_nodes (float):      Average number of nodes per graph
        avg_degree (float):         Average degree of nodes in the graph
    """

    def __init__(
        self,
        use_pbc: bool = True,
        regress_forces: bool = True,
        otf_graph: bool = True,
        max_neighbors: int = 500,
        max_radius: float = 5.0,
        max_num_elements: int = 90,
        xas_feature_dim: int = 200,
        num_layers: int = 12,
        sphere_channels: int = 128,
        attn_hidden_channels: int = 128,
        num_heads: int = 8,
        attn_alpha_channels: int = 32,
        attn_value_channels: int = 16,
        ffn_hidden_channels: int = 512,
        norm_type: str = "rms_norm_sh",
        lmax_list: List[int] = [6],
        mmax_list: List[int] = [2],
        grid_resolution: Optional[int] = None,
        num_sphere_samples: int = 128,
        edge_channels: int = 128,
        use_atom_edge_embedding: bool = True,
        share_atom_edge_embedding: bool = False,
        use_m_share_rad: bool = False,
        distance_function: str = "gaussian",
        num_distance_basis: int = 512,
        attn_activation: str = "scaled_silu",
        use_s2_act_attn: bool = False,
        use_attn_renorm: bool = True,
        ffn_activation: str = "scaled_silu",
        use_gate_act: bool = False,
        use_grid_mlp: bool = False,
        use_sep_s2_act: bool = True,
        alpha_drop: float = 0.1,
        drop_path_rate: float = 0.05,
        proj_drop: float = 0.0,
        weight_init: str = "normal",
        enforce_max_neighbors_strictly: bool = True,
        avg_num_nodes: Optional[float] = None,
        avg_degree: Optional[float] = None,
    ):
        super().__init__()

        import sys

        if "e3nn" not in sys.modules:
            logging.error("You need to install e3nn==0.4.4 to use EquiformerV2.")
            raise ImportError

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        self.max_num_elements = max_num_elements
        self.xas_feature_dim = xas_feature_dim

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop

        self.avg_num_nodes = avg_num_nodes or _AVG_NUM_NODES
        self.avg_degree = avg_degree or _AVG_DEGREE

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly

        self.device = "cpu"  # torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions: int = len(self.lmax_list)
        self.sphere_channels_all: int = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = nn.Embedding(self.max_num_elements, self.sphere_channels_all)

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
        ]
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                0.0,
                self.cutoff,
                600,
                2.0,
            )
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [self.edge_channels] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.target_embedding = nn.Embedding(self.max_num_elements, self.edge_channels_list[-1])
            self.edge_channels_list[0] = self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(f"({max(self.lmax_list)}, {max(self.lmax_list)})")
        for lval in range(max(self.lmax_list) + 1):
            SO3_m_grid = nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        lval,
                        m,
                        resolution=self.grid_resolution,
                        normalization="component",
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=self.avg_degree,
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(
            self.norm_type,
            lmax=max(self.lmax_list),
            num_channels=self.sphere_channels,
        )
        self.xas_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            self.sphere_channels,  # Output XAS features instead of 1 for energy
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )
        
        total_sh_coeffs = sum((l + 1) ** 2 for l in self.lmax_list) * self.num_resolutions
        flattened_size = total_sh_coeffs * self.sphere_channels
        self.xas_projection = nn.Linear(flattened_size, self.xas_feature_dim)
        # Element type embedding for multi-element training (C=0, N=1, O=2)
        self.element_embedding = nn.Embedding(3, flattened_size)
        
        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def _generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
        enforce_max_neighbors_strictly=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if enforce_max_neighbors_strictly is not None:
            pass
        elif hasattr(self, "enforce_max_neighbors_strictly"):
            # Not all models will have this attribute
            enforce_max_neighbors_strictly = self.enforce_max_neighbors_strictly
        else:
            # Default to old behavior
            enforce_max_neighbors_strictly = True

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning("Turning otf_graph=True as required attributes not present in data object")
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data,
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
            cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    @torch.enable_grad()
    def forward(self, data: Data, is_training: bool = True):
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        atomic_numbers = data.z.long()
        num_atoms = len(atomic_numbers)

        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            _,  # cell offset distances
            neighbors,
        ) = self._generate_graph(
            data,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
        )

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge
        edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat((edge_distance, source_embedding, target_embedding), dim=1)

        # Edge-degree embedding
        edge_degree = self.edge_degree_embedding(atomic_numbers, edge_distance, edge_index)
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # XAS spectra prediction
        ###############################################################
        node_xas  = self.xas_block(x)

        node_features = node_xas.embedding.reshape(num_atoms, -1)  # Shape: [num_atoms, num_sh_coeffs * sphere_channels]
        # Element type conditioning for multi-element training
        if hasattr(data, 'element_type') and data.element_type is not None:
            element_emb = self.element_embedding(data.element_type)  # [batch_size, flattened_size]
            element_emb_per_atom = element_emb[data.batch]  # [num_atoms, flattened_size]
            node_features = node_features + element_emb_per_atom
        xas_predictions = self.xas_projection(node_features)  # Shape: [num_atoms, xas_feature_dim]
        
        if is_training:
            xas_target = data.y  # Shape: [num_atoms, xas_feature_dim]
            xas_mask = data.xas_mask  # Shape: [num_atoms], boolean mask
            if xas_mask.sum() == 0:
                # If no atoms have XAS data, return zero loss
                loss = torch.tensor(0.0, device=xas_predictions.device, requires_grad=True)
            else:
                # Apply mask to get only relevant predictions and targets
                masked_pred = xas_predictions[xas_mask]  # Shape: [num_xas_atoms, xas_feature_dim]
                masked_target = xas_target[xas_mask]  # Shape: [num_xas_atoms, xas_feature_dim]
                
                # Calculate loss only for masked atoms
                loss = F.mse_loss(masked_pred, masked_target, reduction='sum')
                
                return loss, {'total_loss': loss.item()}
        else:
            return xas_predictions

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) or isinstance(module, SO3_LinearV2):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)

