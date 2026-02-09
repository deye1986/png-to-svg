"""Path tracing - pure functional approach.

Traces contours from bitmap and converts to vector paths.
"""
import numpy as np
from typing import List, Tuple
from skimage import measure

from ..models import Bitmap, Point, BezierCurve, VectorPath


def trace_paths(bitmap: Bitmap, color_value: int = 255) -> List[VectorPath]:
    """Trace contours from binary bitmap.
    
    Args:
        bitmap: Binary bitmap to trace
        color_value: Pixel value to trace (default 255 for white)
        
    Returns:
        List of vector paths
    """
    # Find contours
    contours = measure.find_contours(bitmap.data, level=color_value / 2)
    
    paths = []
    for contour in contours:
        if len(contour) < 3:
            continue
        
        # Convert contour points to our Point type
        points = [Point(x=float(pt[1]), y=float(pt[0])) for pt in contour]
        
        # Fit Bezier curves to points
        curves = fit_bezier_curves(points)
        
        if curves:
            path = VectorPath(
                curves=tuple(curves),
                color='#000000',
                closed=True
            )
            paths.append(path)
    
    return paths


def trace_multi_color_paths(bitmap: Bitmap) -> List[VectorPath]:
    """Trace paths for each unique color in bitmap.
    
    Args:
        bitmap: Color-quantized bitmap
        
    Returns:
        List of vector paths, one set per color
    """
    paths = []
    
    # Get unique colors
    if bitmap.is_grayscale:
        unique_colors = np.unique(bitmap.data)
    else:
        # Reshape to (pixels, channels) and find unique rows
        pixels = bitmap.data.reshape(-1, bitmap.data.shape[-1])
        unique_colors = np.unique(pixels, axis=0)
    
    # Trace each color separately
    for color in unique_colors:
        # Create binary mask for this color
        if bitmap.is_grayscale:
            mask = (bitmap.data == color).astype(np.uint8) * 255
        else:
            mask = np.all(bitmap.data == color, axis=-1).astype(np.uint8) * 255
        
        # Skip if no pixels
        if not mask.any():
            continue
        
        # Create temporary bitmap
        color_bitmap = Bitmap(
            width=bitmap.width,
            height=bitmap.height,
            data=mask,
            mode='L'
        )
        
        # Trace paths for this color
        color_paths = trace_paths(color_bitmap)
        
        # Set color
        hex_color = _to_hex_color(color) if not bitmap.is_grayscale else f'#{int(color):02x}{int(color):02x}{int(color):02x}'
        
        for path in color_paths:
            # Create new path with correct color (paths are immutable)
            colored_path = VectorPath(
                curves=path.curves,
                color=hex_color,
                closed=path.closed
            )
            paths.append(colored_path)
    
    return paths


def fit_bezier_curves(
    points: List[Point],
    smoothness: float = 0.5
) -> List[BezierCurve]:
    """Fit Bezier curves to a sequence of points.
    
    Uses simplified curve fitting. For production, consider potrace algorithm.
    
    Args:
        points: List of points forming a contour
        smoothness: Smoothing factor (0.0 = angular, 1.0 = smooth)
        
    Returns:
        List of Bezier curves approximating the points
    """
    if len(points) < 3:
        return []
    
    curves = []
    
    # Simplify points first (Douglas-Peucker style)
    simplified = _simplify_points(points, tolerance=1.0 + smoothness * 2.0)
    
    # Create cubic Bezier segments
    n = len(simplified)
    for i in range(n):
        start = simplified[i]
        end = simplified[(i + 1) % n]
        
        # Simple control point estimation
        # For better results, use potrace's corner detection
        prev_point = simplified[(i - 1) % n]
        next_point = simplified[(i + 2) % n]
        
        # Control points along tangent direction
        control1 = Point(
            x=start.x + (end.x - prev_point.x) * smoothness * 0.25,
            y=start.y + (end.y - prev_point.y) * smoothness * 0.25
        )
        control2 = Point(
            x=end.x - (next_point.x - start.x) * smoothness * 0.25,
            y=end.y - (next_point.y - start.y) * smoothness * 0.25
        )
        
        curve = BezierCurve(
            start=start,
            control1=control1,
            control2=control2,
            end=end
        )
        curves.append(curve)
    
    return curves


def _simplify_points(points: List[Point], tolerance: float) -> List[Point]:
    """Simplify point sequence using Douglas-Peucker algorithm.
    
    Args:
        points: Input points
        tolerance: Simplification tolerance
        
    Returns:
        Simplified point list
    """
    if len(points) <= 2:
        return points
    
    # Find point with maximum distance
    max_dist = 0.0
    max_index = 0
    
    for i in range(1, len(points) - 1):
        dist = _perpendicular_distance(points[i], points[0], points[-1])
        if dist > max_dist:
            max_dist = dist
            max_index = i
    
    # If max distance is greater than tolerance, recursively simplify
    if max_dist > tolerance:
        left = _simplify_points(points[:max_index + 1], tolerance)
        right = _simplify_points(points[max_index:], tolerance)
        return left[:-1] + right
    else:
        return [points[0], points[-1]]


def _perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """Calculate perpendicular distance from point to line."""
    dx = line_end.x - line_start.x
    dy = line_end.y - line_start.y
    
    if dx == 0 and dy == 0:
        return np.hypot(point.x - line_start.x, point.y - line_start.y)
    
    t = ((point.x - line_start.x) * dx + (point.y - line_start.y) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    
    proj_x = line_start.x + t * dx
    proj_y = line_start.y + t * dy
    
    return np.hypot(point.x - proj_x, point.y - proj_y)


def _to_hex_color(color) -> str:
    """Convert color array to hex string."""
    if isinstance(color, (int, np.integer)):
        return f'#{int(color):02x}{int(color):02x}{int(color):02x}'
    elif len(color) >= 3:
        return f'#{int(color[0]):02x}{int(color[1]):02x}{int(color[2]):02x}'
    return '#000000'