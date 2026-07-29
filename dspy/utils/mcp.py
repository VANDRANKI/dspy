from typing import TYPE_CHECKING, Any

from dspy.adapters.types.tool import Tool, convert_input_schema_to_tool_args

if TYPE_CHECKING:
    import mcp


def _convert_mcp_tool_result(call_tool_result: "mcp.types.CallToolResult") -> str | list[Any]:
    """Flatten an MCP tool call result into DSPy-friendly content.

    Text content items are collapsed into a single string (or a list of strings when
    there are multiple), while non-text content items are passed through unchanged.

    Args:
        call_tool_result: The raw result returned by `mcp.ClientSession.call_tool`.

    Returns:
        The extracted text (a single string if there was exactly one text content item,
        otherwise a list mixing text strings and any non-text content items).

    Raises:
        RuntimeError: If `call_tool_result.isError` is set, indicating the MCP tool
            call itself failed.
    """
    from mcp.types import TextContent

    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content = [content.text for content in text_contents]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise RuntimeError(f"Failed to call a MCP tool: {tool_content}")

    return tool_content or non_text_contents


def convert_mcp_tool(session: "mcp.ClientSession", tool: "mcp.types.Tool") -> Tool:
    """Build a DSPy tool from an MCP tool.

    Args:
        session: The MCP session to use.
        tool: The MCP tool to convert.

    Returns:
        A dspy Tool object.
    """
    args, arg_types, arg_desc = convert_input_schema_to_tool_args(tool.inputSchema)

    # Convert the MCP tool and Session to a single async method
    async def func(*args, **kwargs):
        result = await session.call_tool(tool.name, arguments=kwargs)
        return _convert_mcp_tool_result(result)

    return Tool(func=func, name=tool.name, desc=tool.description, args=args, arg_types=arg_types, arg_desc=arg_desc)
