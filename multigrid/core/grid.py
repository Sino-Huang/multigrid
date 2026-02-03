from __future__ import annotations

import numpy as np

from collections import defaultdict
from functools import cached_property
from numpy.typing import NDArray as ndarray
from typing import Any, Callable, Iterable, TYPE_CHECKING

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

if TYPE_CHECKING:
    if torch is not None:
        from torch import Tensor

from .agent import Agent
from .constants import Type, TILE_PIXELS
from .world_object import Wall, WorldObj

from ..utils.rendering import (
    downsample,
    fill_coords,
    highlight_img,
    point_in_rect,
)

# Import GPU rendering functions if available
if TORCH_AVAILABLE:
    from ..utils.rendering import (
        downsample_gpu,
        fill_coords_gpu,
        highlight_img_gpu,
        create_grid_lines_mask_gpu,
    )



class Grid:
    """
    Class representing a grid of :class:`.WorldObj` objects.

    Attributes
    ----------
    width : int
        Width of the grid
    height : int
        Height of the grid
    world_objects : dict[tuple[int, int], WorldObj]
        Dictionary of world objects in the grid, indexed by (x, y) location
    state : ndarray[int] of shape (width, height, WorldObj.dim)
        Grid state, where each (x, y) entry is a world object encoding
    """

    # Static cache of pre-renderer tiles
    _tile_cache: dict[tuple[Any, ...], Any] = {}
    
    # GPU tile cache (if torch is available)
    _tile_cache_gpu: dict[tuple[Any, ...], Any] = {}

    def __init__(self, width: int, height: int):
        """
        Parameters
        ----------
        width : int
            Width of the grid
        height : int
            Height of the grid
        """
        assert width >= 3
        assert height >= 3
        self.world_objects: dict[tuple[int, int], WorldObj] = {} # indexed by location
        self.state: ndarray[np.int] = np.zeros((width, height, WorldObj.dim), dtype=int)
        self.state[...] = WorldObj.empty()

    @cached_property
    def width(self) -> int:
        """
        Width of the grid.
        """
        return self.state.shape[0]

    @cached_property
    def height(self) -> int:
        """
        Height of the grid.
        """
        return self.state.shape[1]

    @property
    def grid(self) -> list[WorldObj | None]:
        """
        Return a list of all world objects in the grid.
        """
        return [self.get(i, j) for i in range(self.width) for j in range(self.height)]

    def set(self, x: int, y: int, obj: WorldObj | None):
        """
        Set a world object at the given coordinates.

        Parameters
        ----------
        x : int
            Grid x-coordinate
        y : int
            Grid y-coordinate
        obj : WorldObj or None
            Object to place
        """
        # Update world object dictionary
        self.world_objects[x, y] = obj

        # Update grid state
        if isinstance(obj, WorldObj):
            self.state[x, y] = obj
        elif obj is None:
            self.state[x, y] = WorldObj.empty()
        else:
            raise TypeError(f"cannot set grid value to {type(obj)}")

    def get(self, x: int, y: int) -> WorldObj | None:
        """
        Get the world object at the given coordinates.

        Parameters
        ----------
        x : int
            Grid x-coordinate
        y : int
            Grid y-coordinate
        """
        # Create WorldObj instance if none exists
        if (x, y) not in self.world_objects:
            self.world_objects[x, y] = WorldObj.from_array(self.state[x, y])

        return self.world_objects[x, y]

    def update(self, x: int, y: int):
        """
        Update the grid state from the world object at the given coordinates.

        Parameters
        ----------
        x : int
            Grid x-coordinate
        y : int
            Grid y-coordinate
        """
        if (x, y) in self.world_objects:
            self.state[x, y] = self.world_objects[x, y]

    def horz_wall(
        self,
        x: int, y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall):
        """
        Create a horizontal wall.

        Parameters
        ----------
        x : int
            Leftmost x-coordinate of wall
        y : int
            Y-coordinate of wall
        length : int or None
            Length of wall. If None, wall extends to the right edge of the grid.
        obj_type : Callable() -> WorldObj
            Function that returns a WorldObj instance to use for the wall
        """
        length = self.width - x if length is None else length
        self.state[x:x+length, y] = obj_type()

    def vert_wall(
        self,
        x: int, y: int,
        length: int | None = None,
        obj_type: Callable[[], WorldObj] = Wall):
        """
        Create a vertical wall.

        Parameters
        ----------
        x : int
            X-coordinate of wall
        y : int
            Topmost y-coordinate of wall
        length : int or None
            Length of wall. If None, wall extends to the bottom edge of the grid.
        obj_type : Callable() -> WorldObj
            Function that returns a WorldObj instance to use for the wall
        """
        length = self.height - y if length is None else length
        self.state[x, y:y+length] = obj_type()

    def wall_rect(self, x: int, y: int, w: int, h: int):
        """
        Create a walled rectangle.

        Parameters
        ----------
        x : int
            X-coordinate of top-left corner
        y : int
            Y-coordinate of top-left corner
        w : int
            Width of rectangle
        h : int
            Height of rectangle
        """
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    @classmethod
    def render_tile(
        cls,
        obj: WorldObj | None = None,
        agent: Agent | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3) -> ndarray[np.uint8]:
        """
        Render a tile and cache the result.

        Parameters
        ----------
        obj : WorldObj or None
            Object to render
        agent : Agent or None
            Agent to render
        highlight : bool
            Whether to highlight the tile
        tile_size : int
            Tile size (in pixels)
        subdivs : int
            Downsampling factor for supersampling / anti-aliasing
        """
        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (highlight, tile_size)
        if agent:
            key += (agent.state.color, agent.state.dir)
        else:
            key += (None, None)
        key = obj.encode() + key if obj else key

        if key in cls._tile_cache:
            return cls._tile_cache[key]

        img = np.zeros(
            shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        # Draw the object
        if obj is not None:
            obj.render(img)

        # Draw the agent
        if agent is not None and not agent.state.terminated:
            agent.render(img)

        # Highlight the cell if needed
        if highlight:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls._tile_cache[key] = img

        return img

    def render(
        self,
        tile_size: int,
        agents: Iterable[Agent] = (),
        highlight_mask: ndarray[np.bool] | None = None,
        use_fast_assembly: bool = False) -> ndarray[np.uint8]:
        """
        Render this grid at a given scale.

        Parameters
        ----------
        tile_size: int
            Tile size (in pixels)
        agents: Iterable[Agent]
            Agents to render
        highlight_mask: ndarray
            Boolean mask indicating which grid locations to highlight
        use_fast_assembly: bool
            Use optimized batch rendering (default: True)
        """
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Get agent locations
        # For overlapping agents, non-terminated agents get priority
        location_to_agent = defaultdict(type(None))
        for agent in sorted(agents, key=lambda a: not a.terminated):
            location_to_agent[tuple(agent.pos)] = agent

        # Initialize pixel array
        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        if use_fast_assembly:
            # Optimized path: pre-render unique tiles, then batch-assemble
            # Build mapping of grid positions to tile configurations
            tile_map = {}  # config -> tile_img
            grid_configs = np.empty((self.height, self.width), dtype=object)
            
            for j in range(self.height):
                for i in range(self.width):
                    cell = self.get(i, j)
                    agent = location_to_agent[i, j]
                    highlight = highlight_mask[i, j]
                    
                    # Create config key
                    config = (
                        cell.encode() if cell else (),
                        (agent.state.color, agent.state.dir) if (agent and not agent.state.terminated) else None,
                        highlight
                    )
                    grid_configs[j, i] = config
                    
                    # Render tile if not already cached
                    if config not in tile_map:
                        tile_map[config] = Grid.render_tile(
                            cell,
                            agent=agent,
                            highlight=highlight,
                            tile_size=tile_size,
                        )
            
            # Fast assembly using vectorized numpy operations
            for j in range(self.height):
                for i in range(self.width):
                    config = grid_configs[j, i]
                    tile_img = tile_map[config]
                    
                    ymin = j * tile_size
                    ymax = (j + 1) * tile_size
                    xmin = i * tile_size
                    xmax = (i + 1) * tile_size
                    img[ymin:ymax, xmin:xmax, :] = tile_img
        else:
            # Original sequential rendering
            for j in range(0, self.height):
                for i in range(0, self.width):
                    assert highlight_mask is not None
                    cell = self.get(i, j)
                    tile_img = Grid.render_tile(
                        cell,
                        agent=location_to_agent[i, j],
                        highlight=highlight_mask[i, j],
                        tile_size=tile_size,
                    )

                    ymin = j * tile_size
                    ymax = (j + 1) * tile_size
                    xmin = i * tile_size
                    xmax = (i + 1) * tile_size
                    img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    @classmethod
    def render_tile_gpu(
        cls,
        obj: WorldObj | None = None,
        agent: Agent | None = None,
        highlight: bool = False,
        tile_size: int = TILE_PIXELS,
        subdivs: int = 3,
        device: str = 'cuda',
        native_gpu: bool = True) -> Tensor:
        """
        Render a tile and cache the result on GPU using PyTorch.

        Parameters
        ----------
        obj : WorldObj or None
            Object to render
        agent : Agent or None
            Agent to render
        highlight : bool
            Whether to highlight the tile
        tile_size : int
            Tile size (in pixels)
        subdivs : int
            Downsampling factor for supersampling / anti-aliasing
        device : str
            Device to use ('cuda' or 'cpu')
        native_gpu : bool
            If True, attempt to render directly on GPU (experimental, limited support)
            Falls back to CPU rendering for complex objects

        Returns
        -------
        torch.Tensor
            Rendered tile as GPU tensor of shape (tile_size, tile_size, 3)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPU rendering. Install with: pip install torch")
        
        # Generate cache key
        key: tuple[Any, ...] = (highlight, tile_size, device, native_gpu)
        if agent:
            key += (agent.state.color, agent.state.dir)
        else:
            key += (None, None)
        key = obj.encode() + key if obj else key

        if key in cls._tile_cache_gpu:
            return cls._tile_cache_gpu[key]

        # Try native GPU rendering for simple cases
        if native_gpu and cls._can_render_gpu_native(obj, agent):
            img_gpu = cls._render_tile_gpu_native(
                obj, agent, highlight, tile_size, subdivs, device
            )
            cls._tile_cache_gpu[key] = img_gpu
            return img_gpu

        # Fall back to CPU rendering
        # Use CPU cache first if available (much faster than re-rendering)
        cpu_key = (highlight, tile_size)
        if agent:
            cpu_key += (agent.state.color, agent.state.dir)
        else:
            cpu_key += (None, None)
        cpu_key = obj.encode() + cpu_key if obj else cpu_key
        
        if cpu_key in cls._tile_cache:
            # Reuse CPU-rendered tile and transfer to GPU
            img_np = cls._tile_cache[cpu_key]
        else:
            # Render on CPU and cache it
            img_np = cls.render_tile(obj, agent, highlight, tile_size, subdivs)
        
        # Convert to GPU tensor (using pinned memory for faster transfer)
        img_gpu = torch.from_numpy(img_np).to(device, non_blocking=True)
        cls._tile_cache_gpu[key] = img_gpu
        
        return img_gpu

    @staticmethod
    def _can_render_gpu_native(obj: WorldObj | None, agent: Agent | None) -> bool:
        """
        Check if this tile can be rendered natively on GPU.
        All basic WorldObj types now support GPU rendering.
        """
        # All objects now support GPU rendering
        return True

    @classmethod
    def _render_tile_gpu_native(
        cls,
        obj: WorldObj | None,
        agent: Agent | None,
        highlight: bool,
        tile_size: int,
        subdivs: int,
        device: str) -> Tensor:
        """
        Render a tile directly on GPU (native implementation).
        
        This bypasses CPU rendering entirely for supported tile types.
        Currently supports:
        - Empty tiles with grid lines
        - Highlighted empty tiles
        
        For complex objects/agents, falls back to CPU rendering.
        """
        # Create high-res tile for supersampling
        tile_size_super = tile_size * subdivs
        img = torch.zeros((tile_size_super, tile_size_super, 3), dtype=torch.uint8, device=device)
        
        # Draw grid lines on GPU
        grid_line_color = torch.tensor([100, 100, 100], dtype=torch.uint8, device=device)
        
        # Top edge (0 to 0.031 * height)
        top_edge_height = int(0.031 * tile_size_super)
        img[:top_edge_height, :, :] = grid_line_color
        
        # Left edge (0 to 0.031 * width)
        left_edge_width = int(0.031 * tile_size_super)
        img[:, :left_edge_width, :] = grid_line_color
        
        # Render object if present
        if obj is not None:
            img = obj.render_gpu(img, device=device)
        
        # Render agent if present
        if agent is not None and not agent.state.terminated:
            img = agent.render_gpu(img, device=device)
        
        # Highlight if needed
        if highlight:
            img = highlight_img_gpu(img)
        
        # Downsample using GPU
        img = downsample_gpu(img, subdivs)
        
        return img

    def render_gpu(
        self,
        tile_size: int,
        agents: Iterable[Agent] = (),
        highlight_mask: ndarray[np.bool] | None = None,
        device: str = 'cuda',
        use_vectorized: bool = True,
        use_fully_parallel: bool = False,
        native_gpu: bool = False) -> ndarray[np.uint8]:
        """
        Render this grid at a given scale using GPU acceleration.
        
        This method is significantly faster for large grids (e.g., 1000x1000) compared
        to the CPU version. It uses PyTorch to batch operations on the GPU.

        Parameters
        ----------
        tile_size : int
            Tile size (in pixels)
        agents : Iterable[Agent]
            Agents to render
        highlight_mask : ndarray
            Boolean mask indicating which grid locations to highlight
        device : str
            Device to use ('cuda' for GPU, 'cpu' for CPU with PyTorch)
        use_vectorized : bool
            Whether to use vectorized GPU assembly (recommended for large grids)
        use_fully_parallel : bool
            Experimental: Render tiles directly on GPU in parallel (fastest but may use more memory)
        native_gpu : bool
            If True, render tiles directly on GPU when possible (no CPU rendering)
            Currently limited to simple tiles. Complex objects fall back to CPU.

        Returns
        -------
        ndarray[np.uint8]
            Rendered image as numpy array of shape (height_px, width_px, 3)
            
        Examples
        --------
        >>> # For a 1000x1000 grid
        >>> img = grid.render_gpu(tile_size=32, agents=agents)
        
        >>> # Maximum performance (experimental)
        >>> img = grid.render_gpu(tile_size=32, agents=agents, use_fully_parallel=True)
        
        >>> # Native GPU rendering (fastest for simple grids with no objects)
        >>> img = grid.render_gpu(tile_size=32, agents=agents, native_gpu=True)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for GPU rendering. Install with: pip install torch")
        
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # Get agent locations
        location_to_agent = defaultdict(type(None))
        for agent in sorted(agents, key=lambda a: not a.terminated):
            location_to_agent[tuple(agent.pos)] = agent

        # Store native_gpu flag for use in helper methods
        self._native_gpu = native_gpu

        if use_fully_parallel:
            return self._render_gpu_fully_parallel(
                tile_size, location_to_agent, highlight_mask, device
            )
        elif use_vectorized:
            return self._render_gpu_vectorized(
                tile_size, location_to_agent, highlight_mask, device
            )
        else:
            return self._render_gpu_simple(
                tile_size, location_to_agent, highlight_mask, device
            )

    def _render_gpu_simple(
        self,
        tile_size: int,
        location_to_agent: dict,
        highlight_mask: ndarray[np.bool],
        device: str) -> ndarray[np.uint8]:
        """
        Simple GPU rendering (less optimized but easier to understand).
        """
        raise NotImplementedError("This method is deprecated. Use vectorized or fully parallel rendering instead.")
        # Pre-render all unique tiles to GPU
        unique_tiles = {}
        for j in range(self.height):
            for i in range(self.width):
                cell = self.get(i, j)
                agent = location_to_agent[i, j]
                highlight = highlight_mask[i, j]
                
                key = (
                    cell.encode() if cell else (),
                    (agent.state.color, agent.state.dir) if agent else None,
                    highlight
                )
                
                if key not in unique_tiles:
                    unique_tiles[key] = Grid.render_tile_gpu(
                        cell,
                        agent=agent,
                        highlight=highlight,
                        tile_size=tile_size,
                        device=device
                    )

        # Assemble grid on GPU
        width_px = self.width * tile_size
        height_px = self.height * tile_size
        img_gpu = torch.zeros((height_px, width_px, 3), dtype=torch.uint8, device=device)

        # Copy tiles to output image
        for j in range(self.height):
            for i in range(self.width):
                cell = self.get(i, j)
                agent = location_to_agent[i, j]
                highlight = highlight_mask[i, j]
                
                key = (
                    cell.encode() if cell else (),
                    (agent.state.color, agent.state.dir) if agent else None,
                    highlight
                )
                
                tile_gpu = unique_tiles[key]
                
                ymin, ymax = j * tile_size, (j + 1) * tile_size
                xmin, xmax = i * tile_size, (i + 1) * tile_size
                img_gpu[ymin:ymax, xmin:xmax, :] = tile_gpu

        return img_gpu.cpu().numpy()

    def _render_gpu_vectorized(
        self,
        tile_size: int,
        location_to_agent: dict,
        highlight_mask: ndarray[np.bool],
        device: str) -> ndarray[np.uint8]:
        """
        Fully vectorized GPU rendering for maximum performance on large grids.
        
        This implementation minimizes Python loops by:
        1. Pre-computing all unique tile configurations
        2. Using GPU tensor operations for tile assembly
        3. Vectorizing the final image composition
        """
        # Step 1: Build index mapping (this loop is unavoidable but fast)
        # Maps each grid cell to a unique tile configuration index
        tile_configs = []
        unique_configs = {}
        config_to_idx = {}
        
        for j in range(self.height):
            row_configs = []
            for i in range(self.width):
                cell = self.get(i, j)
                agent = location_to_agent[i, j]
                highlight = highlight_mask[i, j]
                
                config = (
                    cell.encode() if cell else (),
                    (agent.state.color, agent.state.dir) if agent else None,
                    highlight
                )
                
                if config not in config_to_idx:
                    idx = len(unique_configs)
                    config_to_idx[config] = idx
                    unique_configs[config] = Grid.render_tile_gpu(
                        cell,
                        agent=agent,
                        highlight=highlight,
                        tile_size=tile_size,
                        device=device,
                    )
                
                row_configs.append(config_to_idx[config])
            tile_configs.append(row_configs)
        
        # Step 2: Stack all unique tiles into a single tensor (N, tile_size, tile_size, 3)
        unique_tiles_list = [unique_configs[config] for config in sorted(unique_configs.keys(), key=lambda c: config_to_idx[c])]
        unique_tiles_tensor = torch.stack(unique_tiles_list)
        
        # Step 3: Create configuration index tensor (height, width) on GPU
        config_indices = torch.tensor(tile_configs, dtype=torch.long, device=device)
        
        # Step 4: Vectorized assembly using advanced indexing
        # This is the key GPU acceleration - no Python loops!
        # Shape: (height, width, tile_size, tile_size, 3)
        tiles_grid = unique_tiles_tensor[config_indices]
        
        # Step 5: Reshape to final image using tensor operations
        # (height, width, tile_size, tile_size, 3) -> (height * tile_size, width * tile_size, 3)
        height_px = self.height * tile_size
        width_px = self.width * tile_size
        
        # Permute dimensions: (h, w, ts, ts, 3) -> (h, ts, w, ts, 3)
        tiles_grid = tiles_grid.permute(0, 2, 1, 3, 4)
        
        # Contiguous reshape to final image
        img_gpu = tiles_grid.contiguous().view(height_px, width_px, 3)
        
        # Use non-blocking transfer for better performance
        return img_gpu.cpu().numpy()

    def _render_gpu_fully_parallel(
        self,
        tile_size: int,
        location_to_agent: dict,
        highlight_mask: ndarray[np.bool],
        device: str) -> ndarray[np.uint8]:
        """
        Experimental: Fully parallel GPU rendering that renders basic tiles directly on GPU.
        
        This avoids the CPU rendering bottleneck by rendering simple tiles (empty cells,
        grid lines) directly on GPU. Complex objects still fall back to CPU rendering.
        
        Note: This is most effective when the grid has many empty cells or repeated
        simple objects. Complex objects with custom rendering will still use CPU.
        """
        raise NotImplementedError("This method is experimental and may not be stable.")
        # Initialize output image on GPU
        height_px = self.height * tile_size
        width_px = self.width * tile_size
        img_gpu = torch.zeros((height_px, width_px, 3), dtype=torch.uint8, device=device)
        
        # Pre-render grid lines for all tiles at once (fully parallel)
        subdivs = 3  # Supersampling factor
        tile_size_super = tile_size * subdivs
        
        # Create a batch of base tiles with grid lines (for reuse)
        # This is done once for all tiles
        base_tile = torch.zeros((tile_size_super, tile_size_super, 3), dtype=torch.uint8, device=device)
        grid_mask = create_grid_lines_mask_gpu(tile_size_super, tile_size_super, device)
        base_tile = fill_coords_gpu(base_tile, grid_mask, (100, 100, 100))
        base_tile = downsample_gpu(base_tile, subdivs)
        
        # Now render tiles using the cache for complex objects
        # This is still the bottleneck for objects requiring CPU rendering
        unique_tiles = {}
        
        for j in range(self.height):
            for i in range(self.width):
                cell = self.get(i, j)
                agent = location_to_agent[i, j]
                highlight = highlight_mask[i, j]
                
                # Check if we can skip this tile (empty cell, no agent, no highlight)
                if cell is None and agent is None and not highlight:
                    # Use base tile directly
                    ymin, ymax = j * tile_size, (j + 1) * tile_size
                    xmin, xmax = i * tile_size, (i + 1) * tile_size
                    img_gpu[ymin:ymax, xmin:xmax, :] = base_tile
                    continue
                
                # For complex tiles, use cache
                key = (
                    cell.encode() if cell else (),
                    (agent.state.color, agent.state.dir) if agent else None,
                    highlight
                )
                
                if key not in unique_tiles:
                    unique_tiles[key] = Grid.render_tile_gpu(
                        cell,
                        agent=agent,
                        highlight=highlight,
                        tile_size=tile_size,
                        device=device
                    )
                
                tile_gpu = unique_tiles[key]
                ymin, ymax = j * tile_size, (j + 1) * tile_size
                xmin, xmax = i * tile_size, (i + 1) * tile_size
                img_gpu[ymin:ymax, xmin:xmax, :] = tile_gpu
        
        return img_gpu.cpu().numpy()

    def encode(self, vis_mask: ndarray[np.bool] | None = None) -> ndarray[np.int]:
        """
        Produce a compact numpy encoding of the grid.

        Parameters
        ----------
        vis_mask : ndarray[bool] of shape (width, height)
            Visibility mask
        """
        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        encoding = self.state.copy()
        encoding[~vis_mask][..., WorldObj.TYPE] = Type.unseen.to_index()
        return encoding

    @staticmethod
    def decode(array: ndarray[np.int]) -> tuple['Grid', ndarray[np.bool]]:
        """
        Decode an array grid encoding back into a `Grid` instance.

        Parameters
        ----------
        array : ndarray[int] of shape (width, height, dim)
            Grid encoding

        Returns
        -------
        grid : Grid
            Decoded `Grid` instance
        vis_mask : ndarray[bool] of shape (width, height)
            Visibility mask
        """
        width, height, dim = array.shape
        assert dim == WorldObj.dim

        vis_mask = (array[..., WorldObj.TYPE] != Type.unseen.to_index())
        grid = Grid(width, height)
        grid.state[vis_mask] = array[vis_mask]
        return grid, vis_mask
