import os
import mysql.connector

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

mcp = FastMCP("project_database")

def get_connection():

    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE"),
    )

@mcp.tool()
def create_user(
    pw: str,
    name: str,
    gender: str,
    phone: str,
    region: str,
    note: str = "",
) -> str:
    """새로운 회원을 등록한다."""

    conn = get_connection()
    cursor = conn.cursor()

    try:
        sql = """
        INSERT INTO user_Project
        (pw, name, gender, phone, region, note)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        cursor.execute(
            sql,
            (pw, name, gender, phone, region, note)
        )

        conn.commit()

        return f"회원 등록 완료: user_id={cursor.lastrowid}"

    finally:
        cursor.close()
        conn.close()

@mcp.tool()
def get_user(user_id: int) -> dict:
    """user_id로 회원 정보를 조회한다."""

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT
                user_id,
                name,
                gender,
                phone,
                region,
                note
            FROM user_Project
            WHERE user_id = %s
            """,
            (user_id,)
        )

        result = cursor.fetchone()

        return result or {}

    finally:
        cursor.close()
        conn.close()

@mcp.tool()
def update_user(
    user_id: int,
    phone: str,
    region: str,
    note: str,
) -> str:
    """회원의 연락처, 지역, 비고를 수정한다."""

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            UPDATE user_Project
            SET
                phone = %s,
                region = %s,
                note = %s
            WHERE user_id = %s
            """,
            (phone, region, note, user_id)
        )

        conn.commit()

        if cursor.rowcount == 0:
            return "해당 회원을 찾을 수 없습니다."

        return f"user_id {user_id} 회원 수정 완료"

    finally:
        cursor.close()
        conn.close()

@mcp.tool()
def delete_user(user_id: int) -> str:
    """회원 정보를 삭제한다."""

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            DELETE FROM user_Project
            WHERE user_id = %s
            """,
            (user_id,)
        )

        conn.commit()

        if cursor.rowcount == 0:
            return "해당 회원을 찾을 수 없습니다."

        return f"user_id {user_id} 회원 삭제 완료"

    finally:
        cursor.close()
        conn.close()

@mcp.tool()
def create_sale(
    sale_date: str,
    user_id: int,
    product_name: str,
    unit_price: float,
    quantity: int,
) -> str:
    """판매 정보를 등록한다."""

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO sale_Project
            (
                sale_date,
                user_id,
                product_name,
                unit_price,
                quantity
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                sale_date,
                user_id,
                product_name,
                unit_price,
                quantity,
            )
        )

        conn.commit()

        return f"판매 등록 완료: sale_id={cursor.lastrowid}"

    finally:
        cursor.close()
        conn.close()

@mcp.tool()
def get_sale(sale_id: int) -> dict:
    """판매 ID로 판매 정보를 조회한다."""

    conn = get_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT *
            FROM sale_Project
            WHERE sale_id = %s
            """,
            (sale_id,)
        )

        return cursor.fetchone() or {}

    finally:
        cursor.close()
        conn.close()

@mcp.tool()
def update_sale(
    sale_id: int,
    unit_price: float,
    quantity: int,
) -> str:
    """판매 단가와 수량을 수정한다."""

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            UPDATE sale_Project
            SET
                unit_price = %s,
                quantity = %s
            WHERE sale_id = %s
            """,
            (unit_price, quantity, sale_id)
        )

        conn.commit()

        if cursor.rowcount == 0:
            return "해당 판매 기록을 찾을 수 없습니다."

        return f"sale_id {sale_id} 수정 완료"

    finally:
        cursor.close()
        conn.close()

@mcp.tool()
def delete_sale(sale_id: int) -> str:
    """판매 기록을 삭제한다."""

    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            DELETE FROM sale_Project
            WHERE sale_id = %s
            """,
            (sale_id,)
        )

        conn.commit()

        if cursor.rowcount == 0:
            return "해당 판매 기록을 찾을 수 없습니다."

        return f"sale_id {sale_id} 판매 기록 삭제 완료"

    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    mcp.run() 