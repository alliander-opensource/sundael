# SPDX-FileCopyrightText: 2026 Contributors to the sundael project.
# SPDX-License-Identifier: MPL-2.0
"""Module for disaggregration of consumption and generation."""

import importlib.metadata

from sundael.config import PVConfig
from sundael.disaggregation import PVDisaggregator

__version__ = importlib.metadata.version(__package__)


__all__ = ["PVConfig", "PVDisaggregator"]
