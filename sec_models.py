from pydantic import BaseModel
from datetime import date, datetime
from typing import Optional


# Pydantic model for a single filing
class Filing(BaseModel):
    accession_number: str
    form_type: str
    filing_date: date
    period_of_report: date
    file_number: Optional[str] = None
    filing_directory: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class Holding(BaseModel):
    holding_id: int
    issuer_name: str
    title_of_class: str
    shares_or_principal_amount: int
    shares_or_principal_type: str
    value: int
    put_or_call: Optional[str] = None
    investment_discretion: Optional[str] = None
    voting_authority_sole: Optional[int] = None
    voting_authority_shared: Optional[int] = None
    voting_authority_none: Optional[int] = None
    cusip: Optional[str] = None
