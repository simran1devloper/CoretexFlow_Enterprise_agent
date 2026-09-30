"""Service entrypoints.

Each subpackage is one deployable unit. They share a container and a codebase
but scale independently, because their workloads differ: API traffic is
high-frequency and short-lived, agent execution is slow and expensive, and
reporting is data-intensive.
"""
