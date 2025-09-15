import psycopg2
from fastapi import FastAPI, Depends, HTTPException, Query
import os
import uvicorn
import polars as pl
from edgar import set_identity
from contextlib import asynccontextmanager
import asyncio
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from fastapi import HTTPException
import duckdb
from typing import Optional
from psycopg2.extras import RealDictCursor
from fastapi.middleware.cors import CORSMiddleware

from sec_models import Filing, Holding


EDGAR_IDENTITY = os.getenv('EDGAR_IDENTITY', "26b610663e50@company.co.uk")
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', None)

set_identity(EDGAR_IDENTITY)

last_modified = 0
subscribers = set()
current_data = None
db_connections: dict[str, Optional[duckdb.DuckDBPyConnection]] = {}

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

origins_env = os.environ.get("CORS_ORIGINS", "").split(',')
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
            row.get('company_mail_street1'),
            row.get('company_mail_street2'),
            row.get('company_mail_city'),
            row.get('company_mail_state'),
            row.get('company_mail_state_desc'),
            row.get('company_mail_zipcode'),
        )
        business_address = _format_address(
            row.get('company_business_street1'),
            row.get('company_business_street2'),
            row.get('company_business_city'),
            row.get('company_business_state'),
            row.get('company_business_state_desc'),
            row.get('company_business_zipcode'),
        )

        companies.append(
            {
                "cik": str(row.get('cik_number')),
                "company_name": row.get('company_name'),
                "company_phone": row.get('company_phone'),
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
            result.get('company_mail_street1'),
            result.get('company_mail_street2'),
            result.get('company_mail_city'),
            result.get('company_mail_state'),
            result.get('company_mail_state_desc'),
            result.get('company_mail_zipcode'),
        )
        business_address = _format_address(
            result.get('company_business_street1'),
            result.get('company_business_street2'),
            result.get('company_business_city'),
            result.get('company_business_state'),
            result.get('company_business_state_desc'),
            result.get('company_business_zipcode'),
        )

        manager_data = {
            "cik": str(result.get('cik_number')),
            "company_name": result.get('company_name'),
            "company_phone": result.get('company_phone'),
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
        company_id = company['company_id']  # type: ignore

        count_query = "SELECT COUNT(*) FROM filings WHERE company_id = %s"
        db.execute(count_query, (company_id,))
        total_count = db.fetchone()['count']

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
        filings = [Filing(**row) for row in filings_data]

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
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    try:
        # Get the total count of filings for pagination metadata
        count_query = "SELECT COUNT(*) FROM filings"
        db.execute(count_query)
        total_count = db.fetchone()['count']  # type: ignore

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

        # Build the query
        filings_query = """
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
            ORDER BY f.filing_date DESC
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
                "next_offset": offset + limit if has_more else None,
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
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: psycopg2.extensions.cursor = Depends(get_db_cursor),
):
    """
    Retrieve a single holding by accession number
    """

    try:
        query = """
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
            WHERE f.accession_number = %s
            ORDER BY h.value DESC
            LIMIT %s OFFSET %s
        """

        db.execute(query, (accession_number, limit + 1, offset))
        holdings_data = db.fetchall()

        # Format the response using the Pydantic model
        has_more = len(holdings_data) > limit
        if has_more:
            holdings_data.pop()  # Remove the extra record

        # Get total count, but only if pagination requires it.
        # This is the only time we need to run a separate count query.
        total_count = None
        if has_more or offset > 0:
            count_query = "SELECT COUNT(*) FROM holdings h INNER JOIN filings f ON h.filing_id = f.filing_id WHERE f.accession_number = %s"
            db.execute(count_query, (accession_number,))
            total_count = db.fetchone()['count']

        if not holdings_data:
            raise HTTPException(
                status_code=404,
                detail=f"No holdings found for accession number {accession_number}",
            )

        holdings = [Holding(**row) for row in holdings_data]

        return {
            "holdings": holdings,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,  # total_count will be None if only one page exists
                "has_more": has_more,
                "next_offset": offset + limit if has_more else None,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import Body


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
    ).head(100)

    closed_positions_top_100 = closed_positions.sort(
        ["total_value_prev", "total_shares_prev"], descending=True
    ).head(100)

    increased_holdings_top_100 = increased_holdings.sort(
        ["change_in_share", "percent_change"], descending=True
    ).head(100)

    decreased_holdings_top_100 = decreased_holdings.sort(
        [
            "change_in_share",
            "percent_change",
        ],
        descending=False,
    ).head(100)

    # preparation to call openai API
    new_holdings_dict = {
        row['issuer_name_clean']: row['total_shares_latest']
        for row in new_holdings_top_100.to_dicts()
    }
    closed_positions_dict = {
        row['issuer_name_clean_prev']: row['total_shares_prev']
        for row in closed_positions_top_100.to_dicts()
    }
    increased_holdings_dict = {
        row['issuer_name_clean']: {
            k: v
            for k, v in row.items()
            if k not in ["issuer_name_clean", "issuer_name_clean_prev"]
        }
        for row in increased_holdings_top_100.to_dicts()
    }
    decreased_holdings_dict = {
        row['issuer_name_clean']: {
            k: v
            for k, v in row.items()
            if k not in ["issuer_name_clean", "issuer_name_clean_prev"]
        }
        for row in decreased_holdings_top_100.to_dicts()
    }

    new_holding_text = '|'.join([f'{k} {v}' for k, v in new_holdings_dict.items()])
    closed_positions_text = '|'.join(
        [f'{k} {v}' for k, v in closed_positions_dict.items()]
    )
    increased_holdings_text = '|'.join(
        [f'{k} {v}' for k, v in increased_holdings_dict.items()]
    )
    decreased_holdings_text = '|'.join(
        [f'{k} {v}' for k, v in decreased_holdings_dict.items()]
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
        f.period_of_report
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

        if previous_filing['cik_number'] != latest_filing['cik_number']:
            return {"error": "CIK for latest and previous quarters do not match"}

        # Ensure correct ordering based on filing dates
        if previous_filing['period_of_report'] > latest_filing['period_of_report']:
            previous_filing, latest_filing = latest_filing, previous_filing
            acc_prev, acc_latest = (
                previous_filing['accession_number'],
                latest_filing['accession_number'],
            )
        amendment_used = None
        message_parts = []
        if latest_filing['accession_number'] != acc_latest:
            message_parts.append(
                f"The latest filing ({acc_latest}) was replaced by its amendment ({latest_filing['accession_number']}) for the comparison."
            )

        if previous_filing['accession_number'] != acc_prev:
            message_parts.append(
                f"The previous filing ({acc_prev}) was replaced by its amendment ({previous_filing['accession_number']}) for the comparison."
            )

        if message_parts:
            amendment_used = " ".join(message_parts)

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

        # Data cleaning and aggregation
        latest_aggregated = (
            latest_df.with_columns(
                pl.col("issuer_name")
                .str.strip_chars()
                .str.to_uppercase()
                .alias("issuer_name_clean")
            )
            .group_by("issuer_name_clean")
            .agg(
                pl.col("shares_or_principal_amount").sum().alias("total_shares_latest"),
                pl.col("value").sum().alias("total_value_latest"),
            )
        )

        prev_aggregated = (
            previous_df.with_columns(
                pl.col("issuer_name")
                .str.strip_chars()
                .str.to_uppercase()
                .alias("issuer_name_clean")
            )
            .group_by("issuer_name_clean")
            .agg(
                pl.col("shares_or_principal_amount").sum().alias("total_shares_prev"),
                pl.col("value").sum().alias("total_value_prev"),
            )
        )

        # join
        merged_df = latest_aggregated.join(
            prev_aggregated, on="issuer_name_clean", how="full", suffix="_prev"
        )

        # if previously no shares, now has shares -> new holding
        new_holdings = merged_df.filter(pl.col("total_shares_prev").is_null())
        closed_positions = merged_df.filter(pl.col("total_shares_latest").is_null())

        # Common holdings with changes (increase or decreases)
        common_holdings = merged_df.filter(
            pl.col("total_shares_prev").is_not_null()
            & pl.col("total_shares_latest").is_not_null()
        ).with_columns(
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

        # Increases
        increased_holdings = common_holdings.filter(pl.col("change_in_share") > 0)
        decreased_holdings = common_holdings.filter(pl.col("change_in_share") < 0)
        unchanged_holdings = common_holdings.filter(pl.col("change_in_share") == 0)

        # Prepare response data
        response_data = {
            "metadata": {
                "ai_summary": "Not Available",
                "amendment_used": amendment_used,
                "latest_filing": {
                    "accession_number": latest_filing['accession_number'],
                    "filing_date": (
                        latest_filing['filing_date'].isoformat()
                        if latest_filing['filing_date']
                        else None
                    ),
                    "period_of_report": (
                        latest_filing['period_of_report']
                        if latest_filing['period_of_report']
                        else None
                    ),
                    "form_type": latest_filing['form_type'],
                    "user_input": acc_latest,
                },
                "previous_filing": {
                    "accession_number": previous_filing['accession_number'],
                    "filing_date": (
                        previous_filing['filing_date'].isoformat()
                        if previous_filing['filing_date']
                        else None
                    ),
                    "period_of_report": (
                        previous_filing['period_of_report']
                        if previous_filing['period_of_report']
                        else None
                    ),
                    "form_type": previous_filing['form_type'],
                    "user_input": acc_prev,
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
                "new_holdings": _df_to_dict_list(new_holdings),
                "closed_positions": _df_to_dict_list(closed_positions),
                "increased_holdings": _df_to_dict_list(increased_holdings),
                "decreased_holdings": _df_to_dict_list(decreased_holdings),
                "common_holdings": _df_to_dict_list(unchanged_holdings),
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
