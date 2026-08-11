from modelcore.exceptions.provider import ModelCoreError


class ToolError(ModelCoreError):
    """Base error for explicitly registered tool operations."""


class ToolNotFoundError(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


class ToolRoundLimitError(ToolError):
    pass
