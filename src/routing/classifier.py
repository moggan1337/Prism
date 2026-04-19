"""
Content Classifier - Automatic detection of content types.

Classifies input content into appropriate content types (text, image, audio, video, document)
using multiple detection strategies including:
- MIME type detection
- Content signature analysis
- Extension-based detection
- Heuristic analysis
"""

from __future__ import annotations

import base64
import imghdr
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import structlog

from prism.core.config import ContentType


logger = structlog.get_logger(__name__)


class ContentClassifier:
    """
    Intelligent content classifier for multi-modal content.
    
    Uses multiple detection strategies to accurately classify content:
    - MIME type detection for files
    - Content signature analysis for binary content
    - Heuristic analysis for text content
    - Base64 detection and decoding
    
    Example:
        >>> classifier = ContentClassifier()
        >>> content_type = classifier.classify("Hello, world!")
        >>> print(content_type)
        ContentType.TEXT
        >>> 
        >>> # Classify image bytes
        >>> with open("image.png", "rb") as f:
        ...     img_bytes = f.read()
        >>> content_type = classifier.classify(img_bytes)
        >>> print(content_type)
        ContentType.IMAGE
    """
    
    # Common MIME types mapping
    MIME_TYPE_MAP = {
        # Text types
        "text/plain": ContentType.TEXT,
        "text/html": ContentType.TEXT,
        "text/css": ContentType.TEXT,
        "text/javascript": ContentType.TEXT,
        "text/xml": ContentType.TEXT,
        "application/json": ContentType.TEXT,
        "application/xml": ContentType.TEXT,
        "text/csv": ContentType.TEXT,
        "text/markdown": ContentType.TEXT,
        
        # Image types
        "image/png": ContentType.IMAGE,
        "image/jpeg": ContentType.IMAGE,
        "image/gif": ContentType.IMAGE,
        "image/webp": ContentType.IMAGE,
        "image/svg+xml": ContentType.IMAGE,
        "image/bmp": ContentType.IMAGE,
        "image/tiff": ContentType.IMAGE,
        
        # Audio types
        "audio/mpeg": ContentType.AUDIO,
        "audio/wav": ContentType.AUDIO,
        "audio/ogg": ContentType.AUDIO,
        "audio/flac": ContentType.AUDIO,
        "audio/aac": ContentType.AUDIO,
        "audio/mp4": ContentType.AUDIO,
        
        # Video types
        "video/mp4": ContentType.VIDEO,
        "video/mpeg": ContentType.VIDEO,
        "video/webm": ContentType.VIDEO,
        "video/avi": ContentType.VIDEO,
        "video/quicktime": ContentType.VIDEO,
        
        # Document types
        "application/pdf": ContentType.DOCUMENT,
        "application/msword": ContentType.DOCUMENT,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ContentType.DOCUMENT,
        "application/vnd.ms-excel": ContentType.DOCUMENT,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ContentType.DOCUMENT,
        "application/vnd.ms-powerpoint": ContentType.DOCUMENT,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ContentType.DOCUMENT,
    }
    
    # File extensions mapping
    EXTENSION_MAP = {
        # Text files
        ".txt": ContentType.TEXT,
        ".md": ContentType.TEXT,
        ".json": ContentType.TEXT,
        ".xml": ContentType.TEXT,
        ".html": ContentType.TEXT,
        ".css": ContentType.TEXT,
        ".js": ContentType.TEXT,
        ".ts": ContentType.TEXT,
        ".py": ContentType.TEXT,
        ".java": ContentType.TEXT,
        ".cpp": ContentType.TEXT,
        ".c": ContentType.TEXT,
        ".h": ContentType.TEXT,
        ".csv": ContentType.TEXT,
        
        # Image files
        ".png": ContentType.IMAGE,
        ".jpg": ContentType.IMAGE,
        ".jpeg": ContentType.IMAGE,
        ".gif": ContentType.IMAGE,
        ".webp": ContentType.IMAGE,
        ".bmp": ContentType.IMAGE,
        ".tiff": ContentType.IMAGE,
        ".svg": ContentType.IMAGE,
        
        # Audio files
        ".mp3": ContentType.AUDIO,
        ".wav": ContentType.AUDIO,
        ".ogg": ContentType.AUDIO,
        ".flac": ContentType.AUDIO,
        ".aac": ContentType.AUDIO,
        ".m4a": ContentType.AUDIO,
        
        # Video files
        ".mp4": ContentType.VIDEO,
        ".avi": ContentType.VIDEO,
        ".mkv": ContentType.VIDEO,
        ".mov": ContentType.VIDEO,
        ".webm": ContentType.VIDEO,
        ".flv": ContentType.VIDEO,
        
        # Document files
        ".pdf": ContentType.DOCUMENT,
        ".doc": ContentType.DOCUMENT,
        ".docx": ContentType.DOCUMENT,
        ".xls": ContentType.DOCUMENT,
        ".xlsx": ContentType.DOCUMENT,
        ".ppt": ContentType.DOCUMENT,
        ".pptx": ContentType.DOCUMENT,
    }
    
    # Magic bytes signatures
    MAGIC_SIGNATURES = {
        # Images
        b"\x89PNG\r\n\x1a\n": ContentType.IMAGE,
        b"\xff\xd8\xff": ContentType.IMAGE,
        b"GIF87a": ContentType.IMAGE,
        b"GIF89a": ContentType.IMAGE,
        b"RIFF": ContentType.IMAGE,  # WebP starts with RIFF....WEBP
        b"BM": ContentType.IMAGE,  # BMP
        
        # Audio
        b"ID3": ContentType.AUDIO,  # MP3 with ID3
        b"\xff\xfb": ContentType.AUDIO,  # MP3
        b"\xff\xfa": ContentType.AUDIO,  # MP3
        b"RIFF": ContentType.AUDIO,  # WAV starts with RIFF....WAVE
        b"OggS": ContentType.AUDIO,  # OGG
        
        # Video
        b"\x00\x00\x00": ContentType.VIDEO,  # MP4/MOV
        b"RIFF": ContentType.VIDEO,  # AVI
        
        # PDF
        b"%PDF": ContentType.DOCUMENT,
    }
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize the classifier.
        
        Args:
            confidence_threshold: Minimum confidence to return a confident classification
        """
        self.confidence_threshold = confidence_threshold
    
    def classify(self, content: Any) -> ContentType:
        """
        Classify content into a content type.
        
        Uses multiple detection strategies and returns the most confident result.
        
        Args:
            content: Content to classify (str, bytes, dict, Path, file-like, etc.)
            
        Returns:
            ContentType classification
        """
        if content is None:
            return ContentType.UNKNOWN
        
        # Handle different content types
        if isinstance(content, str):
            return self._classify_text(content)
        
        elif isinstance(content, bytes):
            return self._classify_binary(content)
        
        elif isinstance(content, (dict, list)):
            return self._classify_structured(content)
        
        elif isinstance(content, Path):
            return self._classify_path(content)
        
        elif hasattr(content, "read"):
            # File-like object
            return self._classify_file_object(content)
        
        else:
            # Try to convert to string and classify
            try:
                return self._classify_text(str(content))
            except Exception:
                return ContentType.UNKNOWN
    
    def classify_with_confidence(self, content: Any) -> tuple[ContentType, float]:
        """
        Classify content with confidence score.
        
        Args:
            content: Content to classify
            
        Returns:
            Tuple of (ContentType, confidence_score)
        """
        content_type = self.classify(content)
        confidence = self._calculate_confidence(content, content_type)
        return content_type, confidence
    
    def _classify_text(self, content: str) -> ContentType:
        """Classify text content."""
        # Check if it's base64 encoded
        if self._is_base64(content):
            try:
                decoded = base64.b64decode(content)
                return self._classify_binary(decoded)
            except Exception:
                pass
        
        # Check for code patterns
        if self._is_code(content):
            return ContentType.TEXT
        
        # Check for structured data
        if self._is_structured_text(content):
            return ContentType.TEXT
        
        # Default to text
        return ContentType.TEXT
    
    def _classify_binary(self, content: bytes) -> ContentType:
        """Classify binary content."""
        # Check magic bytes
        for signature, content_type in self.MAGIC_SIGNATURES.items():
            if content.startswith(signature):
                # Special handling for RIFF (could be WAV, WebP, or AVI)
                if signature == b"RIFF":
                    return self._classify_riff(content)
                return content_type
        
        # Check for image using imghdr
        detected_type = imghdr.what(None, h=content[:32])
        if detected_type:
            return ContentType.IMAGE
        
        # Check for common patterns
        if len(content) > 0:
            # Check for UTF-8 text
            try:
                decoded = content.decode("utf-8")
                if self._is_code(decoded):
                    return ContentType.TEXT
                if self._is_structured_text(decoded):
                    return ContentType.TEXT
            except UnicodeDecodeError:
                pass
        
        # Try imghdr for other image types
        try:
            image_type = imghdr.what("", content[:32])
            if image_type:
                return ContentType.IMAGE
        except Exception:
            pass
        
        return ContentType.UNKNOWN
    
    def _classify_riff(self, content: bytes) -> ContentType:
        """Classify RIFF container format (WAV, WebP, AVI)."""
        if len(content) < 12:
            return ContentType.UNKNOWN
        
        # WAV: RIFF....WAVE
        if content[8:12] == b"WAVE":
            return ContentType.AUDIO
        
        # WebP: RIFF....WEBP
        if content[8:12] == b"WEBP":
            return ContentType.IMAGE
        
        # AVI: RIFF....AVI 
        if content[8:12] == b"AVI ":
            return ContentType.VIDEO
        
        return ContentType.UNKNOWN
    
    def _classify_structured(self, content: dict | list) -> ContentType:
        """Classify structured data (JSON, dict, list)."""
        return ContentType.TEXT
    
    def _classify_path(self, path: Path) -> ContentType:
        """Classify based on file path."""
        ext = path.suffix.lower()
        
        if ext in self.EXTENSION_MAP:
            return self.EXTENSION_MAP[ext]
        
        # Try MIME type detection
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and mime_type in self.MIME_TYPE_MAP:
            return self.MIME_TYPE_MAP[mime_type]
        
        return ContentType.UNKNOWN
    
    def _classify_file_object(self, file_obj: Any) -> ContentType:
        """Classify file-like object."""
        # Try to read and classify
        try:
            content = file_obj.read(1024)
            if hasattr(file_obj, "seek"):
                file_obj.seek(0)
            return self._classify_binary(content)
        except Exception:
            return ContentType.UNKNOWN
    
    def _is_base64(self, content: str) -> bool:
        """Check if content is base64 encoded."""
        if not content:
            return False
        
        # Remove potential data URI prefix
        if "," in content and ";" in content:
            content = content.split(",", 1)[1]
        
        # Check for valid base64 characters
        base64_pattern = re.compile(r"^[A-Za-z0-9+/]+=*$")
        return bool(base64_pattern.match(content.strip())) and len(content) >= 4
    
    def _is_code(self, content: str) -> bool:
        """Detect if text content is code."""
        code_indicators = [
            r"^\s*(def|class|function|const|let|var|import|export|return|if|for|while)\s",
            r"^\s*#include",
            r"^\s*package\s+\w+",
            r"^\s*public\s+(class|static|void)",
            r"^\s*def\s+\w+\s*\(",
            r"^\s*fn\s+\w+\s*\(",
            r"^\s*func\s+\w+\s*\(",
            r"^\s*async\s+(def|function)",
            r"^\s*@\w+\s*\(?\s*$",  # Decorators
            r"^\s*\{\s*$",  # Code blocks
            r"^\s*\}\s*$",
        ]
        
        for pattern in code_indicators:
            if re.search(pattern, content, re.MULTILINE):
                return True
        
        return False
    
    def _is_structured_text(self, content: str) -> bool:
        """Detect if text content is structured data (JSON, XML, etc.)."""
        # Check for JSON
        if content.strip().startswith(("{", "[")):
            try:
                json.loads(content)
                return True
            except json.JSONDecodeError:
                pass
        
        # Check for XML
        if "<" in content and ">" in content:
            xml_pattern = re.compile(r"<[a-zA-Z][^>]*>")
            if xml_pattern.search(content):
                return True
        
        # Check for YAML-like structure
        if re.search(r"^\w+:\s*\S", content, re.MULTILINE):
            return True
        
        return False
    
    def _calculate_confidence(self, content: Any, content_type: ContentType) -> float:
        """Calculate confidence score for classification."""
        if content_type == ContentType.UNKNOWN:
            return 0.0
        
        if isinstance(content, str):
            # High confidence for text
            if content_type == ContentType.TEXT:
                return 0.95
        
        elif isinstance(content, bytes):
            # Check magic bytes for high confidence
            for signature, sig_type in self.MAGIC_SIGNATURES.items():
                if content.startswith(signature) and sig_type == content_type:
                    return 0.99
        
        return 0.8  # Default confidence
    
    def get_supported_types(self) -> list[ContentType]:
        """Get list of supported content types."""
        return [
            ContentType.TEXT,
            ContentType.IMAGE,
            ContentType.AUDIO,
            ContentType.VIDEO,
            ContentType.DOCUMENT,
        ]
    
    def get_type_metadata(self, content_type: ContentType) -> dict[str, Any]:
        """Get metadata for a content type."""
        metadata = {
            ContentType.TEXT: {
                "description": "Text content including plain text, code, JSON, XML",
                "extensions": list(set(ext for ext, ct in self.EXTENSION_MAP.items() 
                                      if ct == ContentType.TEXT)),
                "mime_types": [mt for mt, ct in self.MIME_TYPE_MAP.items() 
                               if ct == ContentType.TEXT],
            },
            ContentType.IMAGE: {
                "description": "Image content including PNG, JPEG, GIF, WebP",
                "extensions": list(set(ext for ext, ct in self.EXTENSION_MAP.items() 
                                      if ct == ContentType.IMAGE)),
                "mime_types": [mt for mt, ct in self.MIME_TYPE_MAP.items() 
                               if ct == ContentType.IMAGE],
            },
            ContentType.AUDIO: {
                "description": "Audio content including MP3, WAV, OGG, FLAC",
                "extensions": list(set(ext for ext, ct in self.EXTENSION_MAP.items() 
                                      if ct == ContentType.AUDIO)),
                "mime_types": [mt for mt, ct in self.MIME_TYPE_MAP.items() 
                               if ct == ContentType.AUDIO],
            },
            ContentType.VIDEO: {
                "description": "Video content including MP4, AVI, MKV, WebM",
                "extensions": list(set(ext for ext, ct in self.EXTENSION_MAP.items() 
                                      if ct == ContentType.VIDEO)),
                "mime_types": [mt for mt, ct in self.MIME_TYPE_MAP.items() 
                               if ct == ContentType.VIDEO],
            },
            ContentType.DOCUMENT: {
                "description": "Document content including PDF, DOC, XLS, PPT",
                "extensions": list(set(ext for ext, ct in self.EXTENSION_MAP.items() 
                                      if ct == ContentType.DOCUMENT)),
                "mime_types": [mt for mt, ct in self.MIME_TYPE_MAP.items() 
                               if ct == ContentType.DOCUMENT],
            },
        }
        
        return metadata.get(content_type, {})
