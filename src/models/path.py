"""Vector path data models."""
from dataclasses import dataclass
from typing import List, Tuple
from enum import Enum


class PathType(Enum):
    """Type of path segment."""
    MOVE = 'M'
    LINE = 'L'
    CURVE = 'C'
    CLOSE = 'Z'


@dataclass(frozen=True)
class Point:
    """2D point."""
    x: float
    y: float
    
    def __iter__(self):
        """Allow tuple unpacking."""
        return iter((self.x, self.y))


@dataclass(frozen=True)
class BezierCurve:
    """Cubic Bezier curve segment.
    
    Attributes:
        start: Starting point
        control1: First control point
        control2: Second control point
        end: Ending point
    """
    start: Point
    control1: Point
    control2: Point
    end: Point


@dataclass(frozen=True)
class VectorPath:
    """Vector path composed of curves and lines.
    
    Attributes:
        curves: List of Bezier curves
        color: Fill color as hex string (e.g., '#FF0000')
        closed: Whether path is closed
    """
    curves: Tuple[BezierCurve, ...]
    color: str
    closed: bool = True
    
    def to_svg_path_data(self) -> str:
        """Convert to SVG path 'd' attribute string."""
        if not self.curves:
            return ""
        
        path_parts = []
        first_curve = self.curves[0]
        path_parts.append(f"M {first_curve.start.x},{first_curve.start.y}")
        
        for curve in self.curves:
            path_parts.append(
                f"C {curve.control1.x},{curve.control1.y} "
                f"{curve.control2.x},{curve.control2.y} "
                f"{curve.end.x},{curve.end.y}"
            )
        
        if self.closed:
            path_parts.append("Z")
        
        return " ".join(path_parts)


@dataclass(frozen=True)
class VectorImage:
    """Complete vector image representation.
    
    Attributes:
        width: Image width
        height: Image height
        paths: Collection of vector paths
    """
    width: int
    height: int
    paths: Tuple[VectorPath, ...]