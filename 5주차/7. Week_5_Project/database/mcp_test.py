import asyncio

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


PYTHON_PATH = r"C:\miniconda\envs\langchain_up\python.exe"

SERVER_PATH = (
    r"C:\Users\Admin\Desktop\LangChain"
    r"\04. Week_5_Project\database\mcp_server.py"
)


async def main():

    server_params = StdioServerParameters(
        command=PYTHON_PATH,
        args=[SERVER_PATH]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            # MCP 연결 초기화
            await session.initialize()

            print("MCP 서버 연결 성공")

            # 등록된 Tool 확인
            tools = await session.list_tools()

            print("\n등록된 MCP Tools")

            for tool in tools.tools:
                print(f"- {tool.name}")

            result = await session.call_tool(
                "get_user",
                {
                    "user_id": 1
                }
            )

            print(result)


if __name__ == "__main__":
    asyncio.run(main())