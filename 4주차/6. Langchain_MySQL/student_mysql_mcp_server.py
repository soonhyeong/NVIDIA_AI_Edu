import os
import json
import mysql.connector

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


# 환경변수 로드
load_dotenv()

# MySQL 접속 정보
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE"),
}

def get_connection():
    """MySQL Connection 객체를 생성합니다."""

    return mysql.connector.connect(
        **DB_CONFIG
    )


# MCP Server 생성
mcp = FastMCP(
    "Student MySQL MCP Server",
    stateless_http=True,
    json_response=True,
)


# MCP Tool 1: 학생 이름으로 조회
@mcp.tool()
def search_student(name: str) -> str:
    """
    학생 이름으로 MySQL students 테이블에서
    학생 정보를 조회합니다.
    """

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT id, name, age, course
            FROM students
            WHERE name = %s
            """,
            (name,),
        )

        rows = cursor.fetchall()

        if not rows:
            return f"{name} 학생을 MySQL에서 찾을 수 없습니다."

        return json.dumps(rows,
                          ensure_ascii=False,
                        )

    finally:
        cursor.close()
        conn.close()


# MCP Tool 2: 전체 학생 조회
@mcp.tool()
def list_students() -> str:
    """
    MySQL students 테이블의 전체 학생 목록을 조회합니다.
    """

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT id, name, age, course
            FROM students
            ORDER BY id
            """
        )

        rows = cursor.fetchall()

        return json.dumps(rows,
                          ensure_ascii=False,
                        )

    finally:
        cursor.close()
        conn.close()


# MCP Server 실행
# 기본 MCP Endpoint: http://127.0.0.1:8000/mcp
# 실행: python student_mysql_mcp_server.py


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http"
    )
