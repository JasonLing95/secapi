import psycopg2
from fastapi import FastAPI, Depends, HTTPException, Query, Body
import os
import uvicorn
import polars as pl
import asyncio
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from fastapi import HTTPException
from psycopg2.extras import RealDictCursor
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import datetime as dt

from sec_models import Filing


EDGAR_IDENTITY = os.getenv("EDGAR_IDENTITY", "26b610663e50@company.co.uk")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", None)

last_modified = 0
subscribers = set()
current_data = None

if DEEPSEEK_API_KEY:
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

DEFINED_SCHEMA = {
    "cik": pl.String,
    "company": pl.String,
    "filing_date": pl.Date,
    "accession_no": pl.String,
}

### FastAPI app setup ###
app = FastAPI(title="SEC API", version="1.0.0")

origins_env = os.environ.get("CORS_ORIGINS", "").split(",")
allow_origins = [origin.strip() for origin in origins_env if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,  # Your frontend origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


def get_db_connection():
    """
    Establishes and returns a new PostgreSQL database connection.
    """
    try:
        conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "sec"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "password"),
            port=os.getenv("DB_PORT", "5432"),
        )
        return conn
    except psycopg2.OperationalError as e:
        # Raise an exception that FastAPI can handle
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}")


def get_db_cursor():
    """
    Dependency that yields a database cursor and ensures the connection
    is properly managed.
    """
    conn = get_db_connection()
    # Use RealDictCursor to get results as dictionaries
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        yield cursor
    finally:
        cursor.close()
        conn.close()


@app.get("/")
def read_root():
    return {"message": "Welcome to the SEC API"}


def _format_address(
    street1: str,
    street2: str,
    city: str,
    state: str,
    state_desc: str,
    zipcode: str,
) -> str:
    """Format address components into a structured address"""
    # Clean empty values
    street1 = street1 or ""
    street2 = street2 or ""
    city = city or ""
    state = state or ""
    state_desc = state_desc or ""
    zipcode = zipcode or ""

    # Create full address string
    address_parts = []
    if street1:
        address_parts.append(street1)
    if street2:
        address_parts.append(street2)
    if city:
        address_parts.append(city)
    if state:
        address_parts.append(state)
    if state_desc:
        address_parts.append(state_desc)
    if zipcode:
        address_parts.append(zipcode)

    full_address = ", ".join(address_parts)

    return full_address


@app.get("/managers/", response_model=list[dict])
def get_managers(
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
    limit: int = 100,
    offset: int = 0,
):
    """
    Retrieve all managers with pagination
    """
    query = """
            SELECT 
                cik_number,
                company_name, 
                company_phone,
                company_mail_street1,
                company_mail_street2,
                company_mail_city,
                company_mail_state,
                company_mail_state_desc,
                company_zipcode,
                company_business_street1,
                company_business_street2,
                company_business_city,
                company_business_state,
                company_business_state_desc,
                company_business_zipcode
            FROM companies
            LIMIT %s OFFSET %s
        """
    # Execute the query and fetch the results directly into a Pandas DataFrame
    db.execute(query, (limit, offset))
    results = db.fetchall()

    companies = []
    for row in results:
        # Use dictionary keys from RealDictCursor for cleaner access
        mailing_address = _format_address(
            row.get("company_mail_street1"),
            row.get("company_mail_street2"),
            row.get("company_mail_city"),
            row.get("company_mail_state"),
            row.get("company_mail_state_desc"),
            row.get("company_mail_zipcode"),
        )
        business_address = _format_address(
            row.get("company_business_street1"),
            row.get("company_business_street2"),
            row.get("company_business_city"),
            row.get("company_business_state"),
            row.get("company_business_state_desc"),
            row.get("company_business_zipcode"),
        )

        companies.append(
            {
                "cik": str(row.get("cik_number")),
                "company_name": row.get("company_name"),
                "company_phone": row.get("company_phone"),
                "mailing_address": mailing_address,
                "business_address": business_address,
            }
        )

    return companies


@app.get("/managers/{cik}", response_model=dict)
def get_manager(
    cik: str,
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    """
    Retrieve a single manager by CIK from the database
    """

    query = """
        SELECT
            cik_number,
            company_name, 
            company_phone,
            company_mail_street1,
            company_mail_street2,
            company_mail_city,
            company_mail_state,
            company_mail_state_desc,
            company_zipcode,
            company_business_street1,
            company_business_street2,
            company_business_city,
            company_business_state,
            company_business_state_desc,
            company_business_zipcode
        FROM companies
        WHERE cik_number = %s
    """

    try:
        db.execute(query, (cik,))
        result = db.fetchone()

        if not result:
            raise HTTPException(
                status_code=404, detail=f"Manager with CIK {cik} not found"
            )

        mailing_address = _format_address(
            result.get("company_mail_street1"),
            result.get("company_mail_street2"),
            result.get("company_mail_city"),
            result.get("company_mail_state"),
            result.get("company_mail_state_desc"),
            result.get("company_mail_zipcode"),
        )
        business_address = _format_address(
            result.get("company_business_street1"),
            result.get("company_business_street2"),
            result.get("company_business_city"),
            result.get("company_business_state"),
            result.get("company_business_state_desc"),
            result.get("company_business_zipcode"),
        )

        manager_data = {
            "cik": str(result.get("cik_number")),
            "company_name": result.get("company_name"),
            "company_phone": result.get("company_phone"),
            "mailing_address": mailing_address,
            "business_address": business_address,
        }

        return manager_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.get("/managers/{cik}/filings")
def get_manager_filings(
    cik: str,
    limit: int = Query(100, ge=1, le=1000, description="Number of results to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    """
    Retrieve all filings for a specific manager by CIK number
    """

    try:
        db.execute("SELECT company_id FROM companies WHERE cik_number = %s", (cik,))
        company: dict = db.fetchone()  # type: ignore
        if not company:
            raise HTTPException(
                status_code=404, detail=f"Manager with CIK {cik} not found"
            )
        company_id = company["company_id"]  # type: ignore

        count_query = "SELECT COUNT(*) FROM filings WHERE company_id = %s"
        db.execute(count_query, (company_id,))
        total_count = db.fetchone()["count"]  # type: ignore

        if total_count == 0:
            return {
                "filings": [],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": 0,
                    "has_more": False,
                    "next_offset": None,
                },
            }

        filings_query = """
            SELECT
                accession_number,
                form_type,
                filing_date,
                period_of_report,
                file_number,
                filing_directory,
                created_at,
                updated_at
            FROM filings
            WHERE company_id = %s
            ORDER BY filing_date DESC
            LIMIT %s OFFSET %s
        """
        db.execute(filings_query, (company_id, limit, offset))
        filings_data = db.fetchall()

        # Format the response using the Pydantic model
        filings = [Filing(**row) for row in filings_data]  # type: ignore

        has_more = (offset + len(filings)) < total_count

        return {
            "filings": filings,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,
                "has_more": has_more,
                "next_offset": offset + limit if has_more else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.get("/filings/", response_model=dict)
def get_filings(
    limit: int = Query(100, description="Number of items to return", ge=1, le=100),
    offset: int = Query(0, description="Number of items to skip", ge=0),
    sort_by: str = Query("filing_date", description="Column to sort by"),
    sort_order: str = Query("desc", description="Sort order: 'asc' or 'desc'"),
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    allowed_sort_columns = {
        "company_name": "c.company_name",
        "cik_number": "c.cik_number",
        "form_type": "f.form_type",
        "accession_number": "f.accession_number",
        "filing_date": "f.filing_date",
        "period_of_report": "f.period_of_report",
        "created_at": "f.created_at",
    }

    if sort_by not in allowed_sort_columns:
        raise HTTPException(status_code=400, detail="Invalid sort column specified.")

    # --- 2. Security: Validate sort order ---
    if sort_order.lower() not in ["asc", "desc"]:
        raise HTTPException(
            status_code=400, detail="Invalid sort order. Use 'asc' or 'desc'."
        )

    # Get the safe, validated column name and order
    sort_column = allowed_sort_columns[sort_by]

    try:
        # Get the total count of filings for pagination metadata
        count_query = "SELECT COUNT(*) FROM filings"
        db.execute(count_query)
        total_count = db.fetchone()["count"]  # type: ignore

        if total_count == 0:
            return {
                "filings": [],
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "total": 0,
                    "has_more": False,
                },
                "sorting": {
                    "current_sort_by": sort_by,
                    "current_sort_order": sort_order,
                },
            }

        filings_query = f"""
            SELECT
                f.accession_number, f.form_type, f.filing_date, f.period_of_report,
                f.file_number, f.filing_directory, f.created_at, f.updated_at,
                c.company_name, c.cik_number
            FROM filings f
            LEFT JOIN companies c ON f.company_id = c.company_id
            ORDER BY {sort_column} {sort_order.upper()}
            LIMIT %s OFFSET %s
        """

        db.execute(filings_query, (limit, offset))
        filings_data = db.fetchall()

        # Format the response using the Pydantic model
        filings = filings_data

        has_more = (offset + len(filings)) < total_count

        return {
            "filings": filings,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,
                "has_more": has_more,
            },
            "sorting": {
                "current_sort_by": sort_by,
                "current_sort_order": sort_order,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/filings/{accession_number}", response_model=dict)
def get_filing_by_accession(
    accession_number: str,
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):

    try:
        query = """
            SELECT
                f.accession_number,
                f.form_type,
                f.filing_date,
                f.period_of_report,
                f.file_number,
                f.filing_directory,
                f.created_at,
                f.updated_at,
                c.company_name,
                c.cik_number
            FROM filings f
            LEFT JOIN companies c ON f.company_id = c.company_id
            WHERE f.accession_number = %s
        """
        db.execute(query, (accession_number,))
        result = db.fetchone()

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Filing with accession number {accession_number} not found",
            )

        return result  # type: ignore

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/holdings/{accession_number}", response_model=dict)
def get_holding_by_accession_number(
    accession_number: str,
    # limit: int = Query(100, ge=1, le=1000),
    # offset: int = Query(0, ge=0),
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
    draw: int = Query(0, alias="draw"),
    start: int = Query(0, alias="start"),
    length: int = Query(10, alias="length"),
    search_value: Optional[str] = Query(None, alias="search[value]"),
    order_column_index: int = Query(0, alias="order[0][column]"),
    order_dir: str = Query("asc", alias="order[0][dir]"),
):
    """
    Retrieve a single holding by accession number
    """

    try:
        # Map DataTables column index to actual database column names
        column_map = {
            0: "i.issuer_name",
            1: "i.cusip",
            2: "t.name",
            3: "h.value",
            4: "h.shares_or_principal_amount",
            5: "s.name",
            6: "d.name",
            7: "p.name",
            8: "h.voting_authority_sole",
            9: "h.voting_authority_shared",
            10: "h.voting_authority_none",
        }
        order_by_column = column_map.get(order_column_index, "h.value")
        order_direction = "DESC" if order_dir == "desc" else "ASC"

        # Base query to get the total count of all records (before any filtering)
        count_query = """
            SELECT COUNT(*)
            FROM holdings h
            INNER JOIN filings f ON h.filing_id = f.filing_id
            WHERE f.accession_number = %s;
        """
        db.execute(count_query, (accession_number,))
        total_records = db.fetchone()["count"]  # type: ignore

        # Construct the WHERE clause for searching
        where_clause = "WHERE f.accession_number = %s"
        search_params = [accession_number]
        if search_value:
            where_clause += """
                AND (
                    i.issuer_name ILIKE %s OR
                    i.cusip ILIKE %s OR
                    t.name ILIKE %s
                )
            """
            search_pattern = f"%{search_value}%"
            search_params.extend([search_pattern, search_pattern, search_pattern])

        # Query to get the count after applying the search filter
        filtered_count_query = f"""
            SELECT COUNT(*)
            FROM holdings h
            INNER JOIN filings f ON h.filing_id = f.filing_id
            LEFT JOIN issuers i ON h.issuer_id = i.issuer_id
            LEFT JOIN title_of_class_table t ON h.title_of_class = t.id
            {where_clause};
        """
        db.execute(filtered_count_query, search_params)
        filtered_records = db.fetchone()["count"]

        # Main query to fetch the paginated, filtered, and sorted data
        holdings_query = f"""
            SELECT
                h.holding_id,
                i.issuer_name,
                t.name AS title_of_class,
                h.shares_or_principal_amount,
                s.name AS shares_or_principal_type,
                h.value,
                p.name AS put_or_call,
                d.name AS investment_discretion,
                h.voting_authority_sole,
                h.voting_authority_shared,
                h.voting_authority_none,
                i.cusip
            FROM holdings h
            INNER JOIN filings f ON h.filing_id = f.filing_id
            LEFT JOIN issuers i ON h.issuer_id = i.issuer_id
            LEFT JOIN title_of_class_table t ON h.title_of_class = t.id
            LEFT JOIN share_type_table s ON h.shares_or_principal_type = s.id
            LEFT JOIN put_or_call_table p ON h.put_or_call = p.id
            LEFT JOIN investment_discretion_table d ON h.investment_discretion = d.id
            {where_clause}
            ORDER BY {order_by_column} {order_direction}
            OFFSET %s LIMIT %s;
        """
        query_params = search_params + [start, length]
        db.execute(holdings_query, query_params)
        holdings_data = db.fetchall()

        formatted_data = [
            {
                "issuer_name": row.get("issuer_name"),
                "cusip": row.get("cusip"),
                "title_of_class": row.get("title_of_class"),
                "value": row.get("value"),
                "shares_or_principal_amount": row.get("shares_or_principal_amount"),
                "shares_or_principal_type": row.get("shares_or_principal_type"),
                "investment_discretion": row.get("investment_discretion"),
                "put_or_call": row.get("put_or_call"),
                "voting_authority_sole": row.get("voting_authority_sole"),
                "voting_authority_shared": row.get("voting_authority_shared"),
                "voting_authority_none": row.get("voting_authority_none"),
            }
            for row in holdings_data
        ]

        # Format the response for DataTables
        return {
            "draw": draw,
            "recordsTotal": total_records,
            "recordsFiltered": filtered_records,
            "data": formatted_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


class HoldingsRequest(BaseModel):
    new_holdings: List[Dict[str, Any]]
    closed_positions: List[Dict[str, Any]]
    increased_holdings: List[Dict[str, Any]]
    decreased_holdings: List[Dict[str, Any]]


@app.post("/api/ai_summary")
async def openai_call(request: HoldingsRequest = Body(...)):

    if not DEEPSEEK_API_KEY:
        return JSONResponse(
            content={"summary": "API key not configured"}, status_code=503
        )

    # Frontend POST request will be in dict format?
    new_holdings = pl.DataFrame(request.new_holdings)
    closed_positions = pl.DataFrame(request.closed_positions)
    increased_holdings = pl.DataFrame(request.increased_holdings)
    decreased_holdings = pl.DataFrame(request.decreased_holdings)

    new_holdings_top_100 = new_holdings.sort(
        ["total_value_latest", "total_shares_latest"],
        descending=True,
    ).head(5)

    closed_positions_top_100 = closed_positions.sort(
        ["total_value_prev", "total_shares_prev"], descending=True
    ).head(5)

    increased_holdings_top_100 = increased_holdings.sort(
        ["change_in_share", "percent_change"], descending=True
    ).head(5)

    decreased_holdings_top_100 = decreased_holdings.sort(
        [
            "change_in_share",
            "percent_change",
        ],
        descending=False,
    ).head(5)

    # preparation to call openai API
    new_holdings_dict = {
        row["issuer_name_clean"]: row["total_shares_latest"]
        for row in new_holdings_top_100.to_dicts()
    }
    closed_positions_dict = {
        row["issuer_name_clean_prev"]: row["total_shares_prev"]
        for row in closed_positions_top_100.to_dicts()
    }
    increased_holdings_dict = {
        row["issuer_name_clean"]: {
            k: v
            for k, v in row.items()
            if k not in ["issuer_name_clean", "issuer_name_clean_prev"]
        }
        for row in increased_holdings_top_100.to_dicts()
    }
    decreased_holdings_dict = {
        row["issuer_name_clean"]: {
            k: v
            for k, v in row.items()
            if k not in ["issuer_name_clean", "issuer_name_clean_prev"]
        }
        for row in decreased_holdings_top_100.to_dicts()
    }

    new_holding_text = "|".join([f"{k} {v}" for k, v in new_holdings_dict.items()])
    closed_positions_text = "|".join(
        [f"{k} {v}" for k, v in closed_positions_dict.items()]
    )
    increased_holdings_text = "|".join(
        [f"{k} {v}" for k, v in increased_holdings_dict.items()]
    )
    decreased_holdings_text = "|".join(
        [f"{k} {v}" for k, v in decreased_holdings_dict.items()]
    )

    async def get_summary(title, data_text):
        prompt = f"""
        Generate a summary of the holdings changes for the fund management in one or two sentences.
        {title}: {data_text}
        """
        # Assuming an async OpenAI client
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional financial analyst. Be concise.",
                },
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )
        return response.choices[0].message.content

    api_calls = [
        get_summary("New Holdings", new_holding_text),
        get_summary("Closed Positions", closed_positions_text),
        get_summary("Increased Holdings", increased_holdings_text),
        get_summary("Decreased Holdings", decreased_holdings_text),
    ]

    texts = await asyncio.gather(*api_calls)

    final_summary = " ".join([text for text in texts if text])
    return JSONResponse(content={"summary": final_summary})


def _df_to_dict_list(df: pl.DataFrame):
    """Convert DataFrame to list of dictionaries"""
    if df.is_empty():
        return []
    return df.to_dicts()


def run_in_threadpool(func, *args):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, func, *args)


FILING_QUERY_WITH_PRIORITY = """
    SELECT
        f.accession_number,
        c.cik_number,
        f.form_type,
        f.filing_date,
        f.period_of_report,
        f.filing_directory
    FROM filings f
    JOIN companies c ON f.company_id = c.company_id
    WHERE
        c.cik_number = (SELECT c2.cik_number FROM filings f2 JOIN companies c2 ON f2.company_id = c2.company_id WHERE f2.accession_number = %s LIMIT 1)
    AND
        f.period_of_report = (SELECT f3.period_of_report FROM filings f3 WHERE f3.accession_number = %s LIMIT 1)
    ORDER BY
    CASE
        WHEN f.form_type = '13F-HR/A/A' THEN 1
        WHEN f.form_type = '13F-HR/A' THEN 2
        WHEN f.form_type = '13F-HR' THEN 3
        ELSE 4
    END,
    f.filing_date DESC
    LIMIT 1
"""

HOLDINGS_QUERY = """
    SELECT
        h.holding_id,
        i.issuer_name,
        t.name AS title_of_class,
        h.shares_or_principal_amount,
        s.name AS shares_or_principal_type,
        h.value,
        p.name AS put_or_call,
        d.name AS investment_discretion,
        h.voting_authority_sole,
        h.voting_authority_shared,
        h.voting_authority_none,
        i.cusip
    FROM holdings h
    LEFT JOIN issuers i ON h.issuer_id = i.issuer_id
    LEFT JOIN title_of_class_table t ON h.title_of_class = t.id
    LEFT JOIN share_type_table s ON h.shares_or_principal_type = s.id
    LEFT JOIN put_or_call_table p ON h.put_or_call = p.id
    LEFT JOIN investment_discretion_table d ON h.investment_discretion = d.id
    WHERE h.filing_id = (SELECT filing_id FROM filings WHERE accession_number = %s)
    ORDER BY h.value DESC
"""

COMMON_STOCK_TITLE_OF_CLASS = "COM|CL A|COMMON STOCK|STOCK"

MAX_PER_SHARE_PRICE = 11.0
MIN_PER_SHARE_PRICE = 0.01


def _process_filings(
    df: pl.DataFrame,
    latest=True,
):
    suffix = "latest" if latest else "prev"

    df_with_prices = df.with_columns(
        pl.col("value").fill_null(0).alias("value_filled"),
        pl.col("shares_or_principal_amount").fill_null(0).alias("shares_filled"),
    ).with_columns(
        pl.when(pl.col("shares_filled") > 0)
        .then(pl.col("value_filled") / pl.col("shares_filled"))
        .otherwise(pl.lit(0))
        .alias("per_share_price")
    )

    # Get the maximum per-share price from the entire DataFrame
    max_price = df_with_prices["per_share_price"].max()
    min_price = df_with_prices["per_share_price"].min()

    # Determine if all values need to be multiplied
    # If the max price is less than $11, it indicates an error in the whole file
    requires_multiplication = (max_price < MAX_PER_SHARE_PRICE) and (
        min_price < MIN_PER_SHARE_PRICE
    )

    updated_df = (
        df_with_prices.filter(
            pl.col("title_of_class")
            .str.to_uppercase()
            .str.contains(COMMON_STOCK_TITLE_OF_CLASS)
        )
        .with_columns(
            pl.col("issuer_name")
            .str.strip_chars()
            .str.to_uppercase()
            .alias("issuer_name_clean")
        )
        .with_columns(
            # Apply the conditional logic to every row
            pl.when(pl.lit(requires_multiplication))
            .then(pl.col("value_filled") * 1000)
            .otherwise(pl.col("value_filled"))
            .alias("corrected_value")
        )
        .group_by("issuer_name_clean")
        .agg(
            pl.col("shares_or_principal_amount").sum().alias(f"total_shares_{suffix}"),
            pl.col("corrected_value").sum().alias(f"total_value_{suffix}"),
        )
        .with_columns(
            pl.when(pl.col(f"total_shares_{suffix}") > 0)
            .then(pl.col(f"total_value_{suffix}") / pl.col(f"total_shares_{suffix}"))
            .otherwise(pl.lit(0))
            .round(2)
            .alias(f"per_share_price_{suffix}")
        )
    )

    return updated_df, requires_multiplication


@app.get("/analysis/{previous_accession}/{latest_accession}", response_model=dict)
async def compare_holdings(
    previous_accession: str,
    latest_accession: str,
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    """
    Compare two holdings by their accession numbers
    """

    try:

        acc_latest = latest_accession.strip()
        acc_prev = previous_accession.strip()

        # Get filing details for the previous and latest filings
        db.execute(FILING_QUERY_WITH_PRIORITY, (acc_prev, acc_prev))
        previous_filing = db.fetchone()

        db.execute(FILING_QUERY_WITH_PRIORITY, (acc_latest, acc_latest))
        latest_filing = db.fetchone()

        # TODO: fetch filings by API (async)

        if not previous_filing:
            raise HTTPException(
                status_code=404, detail=f"Previous filing {acc_prev} not found"
            )
        if not latest_filing:
            raise HTTPException(
                status_code=404, detail=f"Latest filing {acc_latest} not found"
            )

        latest_acc = latest_filing["accession_number"]  # type: ignore
        previous_acc = previous_filing["accession_number"]  # type: ignore

        if previous_filing["cik_number"] != latest_filing["cik_number"]:
            return {"error": "CIK for latest and previous quarters do not match"}

        # Ensure correct ordering based on filing dates
        if previous_filing["period_of_report"] > latest_filing["period_of_report"]:
            previous_filing, latest_filing = latest_filing, previous_filing
            acc_prev, acc_latest = (
                previous_acc,
                latest_acc,
            )
        amendment_used = None
        message_parts = []
        if latest_acc != acc_latest:
            message_parts.append(
                f"The latest filing ({acc_latest}) was replaced by its amendment ({latest_acc}) for the comparison."
            )

        if previous_acc != acc_prev:
            message_parts.append(
                f"The previous filing ({acc_prev}) was replaced by its amendment ({previous_acc}) for the comparison."
            )

        if message_parts:
            amendment_used = " ".join(message_parts)

        api_save_comparison(
            previous_acc,
            latest_acc,
            db,
        )

        # Fetch holdings data from the database
        db.execute(HOLDINGS_QUERY, (acc_prev,))
        previous_holdings_data = db.fetchall()

        db.execute(HOLDINGS_QUERY, (acc_latest,))
        latest_holdings_data = db.fetchall()

        # Convert to Polars DataFrames for efficient analysis
        if not previous_holdings_data:
            previous_df = pl.DataFrame()
        else:
            previous_df = pl.DataFrame(previous_holdings_data)

        if not latest_holdings_data:
            latest_df = pl.DataFrame()
        else:
            latest_df = pl.DataFrame(latest_holdings_data)

        # TODO: deal with empty holdings

        # Data cleaning and aggregation
        latest_aggregated, latest_multiplication = _process_filings(latest_df)
        prev_aggregated, prev_multiplication = _process_filings(
            previous_df, latest=False
        )

        # TODO: if no holding, issuer_name clean raise error

        # other securities (non-common stock)
        latest_other_securities = latest_df.filter(
            ~pl.col("title_of_class")
            .str.to_uppercase()
            .str.contains(COMMON_STOCK_TITLE_OF_CLASS)
        )

        prev_other_securities = previous_df.filter(
            ~pl.col("title_of_class")
            .str.to_uppercase()
            .str.contains(COMMON_STOCK_TITLE_OF_CLASS)
        )

        # Data cleaning and aggregation for latest other securities
        latest_other_aggregated = (
            latest_other_securities.with_columns(
                pl.col("issuer_name")
                .str.strip_chars()
                .str.to_uppercase()
                .alias("issuer_name_clean")
            )
            .group_by("issuer_name_clean")
            .agg(
                # Renaming the column to avoid confusion with common stock shares
                pl.col("shares_or_principal_amount").sum().alias("total_units_latest"),
                pl.col("value").sum().alias("total_value_latest"),
            )
            # .with_columns(
            #     (pl.col("total_value_latest") * 1000).alias("total_value_latest")
            # )
        )

        # Data cleaning and aggregation for previous other securities
        prev_other_aggregated = (
            prev_other_securities.with_columns(
                pl.col("issuer_name")
                .str.strip_chars()
                .str.to_uppercase()
                .alias("issuer_name_clean")
            )
            .group_by("issuer_name_clean")
            .agg(
                # Renaming the column to avoid confusion with common stock shares
                pl.col("shares_or_principal_amount").sum().alias("total_units_prev"),
                pl.col("value").sum().alias("total_value_prev"),
            )
            # .with_columns((pl.col("total_value_prev") * 1000).alias("total_value_prev"))
        )

        # join
        merged_df = latest_aggregated.join(
            prev_aggregated, on="issuer_name_clean", how="full", suffix="_prev"
        )

        merged_other_df = latest_other_aggregated.join(
            prev_other_aggregated, on="issuer_name_clean", how="full", suffix="_prev"
        )

        # if previously no shares, now has shares -> new holding
        new_holdings = merged_df.filter(pl.col("total_shares_prev").is_null()).select(
            "issuer_name_clean",
            "total_shares_latest",
            "total_value_latest",
            "per_share_price_latest",
        )
        closed_positions = merged_df.filter(
            pl.col("total_shares_latest").is_null()
        ).select(
            "issuer_name_clean_prev",
            "total_shares_prev",
            "total_value_prev",
            "per_share_price_prev",
        )
        new_other_holdings = merged_other_df.filter(
            pl.col("total_units_prev").is_null()
        ).select(
            "issuer_name_clean",
            "total_units_latest",
            "total_value_latest",
        )
        closed_other_positions = merged_other_df.filter(
            pl.col("total_units_latest").is_null()
        ).select(
            "issuer_name_clean_prev",
            "total_units_prev",
            "total_value_prev",
        )

        # Common holdings with changes (increase or decreases)
        common_holdings = (
            merged_df.filter(
                pl.col("total_shares_prev").is_not_null()
                & pl.col("total_shares_latest").is_not_null()
            )
            .with_columns(
                [
                    (pl.col("total_shares_latest") - pl.col("total_shares_prev")).alias(
                        "change_in_share"
                    ),
                    (
                        (pl.col("total_shares_latest") - pl.col("total_shares_prev"))
                        / pl.col("total_shares_prev")
                        * 100
                    )
                    .round(2)
                    .alias("percent_change"),
                ]
            )
            .with_columns(
                pl.when(pl.col("percent_change").is_nan())
                .then(None)
                .when(pl.col("percent_change").is_infinite())
                .then(None)
                .otherwise(pl.col("percent_change"))
                .alias("percent_change")
            )
            .select(
                "issuer_name_clean",
                "total_shares_prev",
                "total_value_prev",
                "per_share_price_prev",
                "total_shares_latest",
                "total_value_latest",
                "per_share_price_latest",
                "change_in_share",
                "percent_change",
            )
        )

        # Increases
        increased_holdings = common_holdings.filter(pl.col("change_in_share") > 0)
        decreased_holdings = common_holdings.filter(pl.col("change_in_share") < 0)
        unchanged_holdings = common_holdings.filter(pl.col("change_in_share") == 0)

        # other securities with changes (increase or decreases)
        common_other_holdings = (
            merged_other_df.filter(
                pl.col("total_units_prev").is_not_null()
                & pl.col("total_units_latest").is_not_null()
            )
            .with_columns(
                [
                    (pl.col("total_units_latest") - pl.col("total_units_prev")).alias(
                        "change_in_units"
                    ),
                    (
                        (pl.col("total_units_latest") - pl.col("total_units_prev"))
                        / pl.col("total_units_prev")
                        * 100
                    )
                    .round(2)
                    .alias("percent_change"),
                ]
            )
            .with_columns(
                pl.when(pl.col("percent_change").is_nan())
                .then(None)
                .when(pl.col("percent_change").is_infinite())
                .then(None)
                .otherwise(pl.col("percent_change"))
                .alias("percent_change")
            )
            .select(
                "issuer_name_clean",
                "total_units_prev",
                "total_value_prev",
                "total_units_latest",
                "total_value_latest",
                "change_in_units",
                "percent_change",
            )
        )

        # Increases
        increased_other_holdings = common_other_holdings.filter(
            pl.col("change_in_units") > 0
        )
        decreased_other_holdings = common_other_holdings.filter(
            pl.col("change_in_units") < 0
        )
        unchanged_other_holdings = common_other_holdings.filter(
            pl.col("change_in_units") == 0
        )

        top_5_new_common = new_holdings.sort(
            by="total_value_latest", descending=True
        ).head(5)
        top_5_closed_common = closed_positions.sort(
            by="total_value_prev", descending=True
        ).head(5)
        top_5_increased_common = increased_holdings.sort(
            by="percent_change", descending=True
        ).head(5)
        top_5_decreased_common = decreased_holdings.sort(by="percent_change").head(5)

        # Prepare response data
        response_data = {
            "metadata": {
                "cik": latest_filing.get("cik_number"),
                "ai_summary": "Not Available",
                "amendment_used": amendment_used,
                "latest_filing": {
                    "accession_number": latest_filing.get("accession_number"),
                    "filing_date": (
                        latest_filing.get("filing_date").isoformat()
                        if latest_filing.get("filing_date")
                        else None
                    ),
                    "period_of_report": latest_filing.get("period_of_report"),
                    "form_type": latest_filing.get("form_type"),
                    "user_input": acc_latest,
                    "filing_directory": latest_filing.get("filing_directory"),
                    "multiplication_applied": latest_multiplication,
                },
                "previous_filing": {
                    "accession_number": previous_filing.get("accession_number"),
                    "filing_date": (
                        previous_filing.get("filing_date").isoformat()
                        if previous_filing.get("filing_date")
                        else None
                    ),
                    "period_of_report": previous_filing.get("period_of_report"),
                    "form_type": previous_filing.get("form_type"),
                    "user_input": acc_prev,
                    "filing_directory": previous_filing.get("filing_directory"),
                    "multiplication_applied": prev_multiplication,
                },
                "summary": {
                    "total_companies_latest": latest_aggregated.height,
                    "total_companies_previous": prev_aggregated.height,
                    "new_holdings_count": new_holdings.height,
                    "closed_positions_count": closed_positions.height,
                    "increased_holdings_count": increased_holdings.height,
                    "decreased_holdings_count": decreased_holdings.height,
                    "unchanged_holdings_count": unchanged_holdings.height,
                },
            },
            "holdings": {
                "new_holdings": {
                    "top_5": top_5_new_common.to_dicts(),
                    "common_stock": _df_to_dict_list(new_holdings),
                    "other_securities": _df_to_dict_list(new_other_holdings),
                },
                "closed_positions": {
                    "top_5": top_5_closed_common.to_dicts(),
                    "common_stock": _df_to_dict_list(closed_positions),
                    "other_securities": _df_to_dict_list(closed_other_positions),
                },
                "increased_holdings": {
                    "top_5": top_5_increased_common.to_dicts(),
                    "common_stock": _df_to_dict_list(increased_holdings),
                    "other_securities": _df_to_dict_list(increased_other_holdings),
                },
                "decreased_holdings": {
                    "top_5": top_5_decreased_common.to_dicts(),
                    "common_stock": _df_to_dict_list(decreased_holdings),
                    "other_securities": _df_to_dict_list(decreased_other_holdings),
                },
                "common_holdings": {
                    "common_stock": _df_to_dict_list(unchanged_holdings),
                    "other_securities": _df_to_dict_list(unchanged_other_holdings),
                },
            },
        }

        return response_data

    except HTTPException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@app.get("/comparisons", response_model=list)
def get_recent_comparisons(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    try:
        query = """
            SELECT accession_number_1, accession_number_2, created_at
            FROM recent_comparisons
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        db.execute(query, (limit, offset))
        comparisons = db.fetchall()
        return comparisons
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ComparisonRequest(BaseModel):
    acc_num_1: str
    acc_num_2: str


@app.post("/comparisons")
def save_comparison(
    request: ComparisonRequest,
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    try:
        query = """
            INSERT INTO recent_comparisons (accession_number_1, accession_number_2)
            VALUES (%s, %s)
        """
        db.execute(query, (request.acc_num_1, request.acc_num_2))
        db.connection.commit()

        return {"message": "Comparison saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def api_save_comparison(accession_1: str, accession_2: str, db):
    try:
        query = """
            INSERT INTO recent_comparisons (accession_number_1, accession_number_2)
            VALUES (%s, %s)
        """
        db.execute(query, (accession_1, accession_2))
        db.connection.commit()

        return {"message": "Comparison saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/company/{cik}/compare/latest")
async def compare_latest_filings(
    cik: str,
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    """
    Automatically compares the latest two quarterly filings for a given company CIK.
    """
    try:
        # Step 1: Find the latest two filings for the given CIK
        filing_query = """
            SELECT
                accession_number,
                period_of_report
            FROM (
                SELECT
                    f.accession_number,
                    f.period_of_report,
                    ROW_NUMBER() OVER(PARTITION BY f.period_of_report ORDER BY f.filing_date DESC, f.accession_number DESC) as rn
                FROM filings f
                INNER JOIN companies c ON f.company_id = c.company_id
                WHERE c.cik_number = %s AND f.form_type IN ('13F-HR', '13F-HR/A', '13F-HR/A/A')
            ) AS ranked_filings
            WHERE rn = 1
            ORDER BY period_of_report desc
            LIMIT 2;
        """
        db.execute(filing_query, (cik,))
        filings = db.fetchall()

        if len(filings) < 2:
            raise HTTPException(
                status_code=404,
                detail=f"Not enough filings found for CIK {cik} to perform a comparison.",
            )

        latest_filing = filings[0]["accession_number"]
        previous_filing = filings[1]["accession_number"]

        print(f"Comparing filings {previous_filing} and {latest_filing} for CIK {cik}")

        # Step 2: Use the existing compare_holdings logic to perform the comparison
        # You'll need to call the function directly here
        return await compare_holdings(
            previous_accession=previous_filing,
            latest_accession=latest_filing,
            db=db,
        )

    except HTTPException:
        raise  # Re-raise HTTPException to be handled by FastAPI
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
