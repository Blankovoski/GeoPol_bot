create extension if not exists vector;

create table if not exists historical_events (
event_id uuid primary key default gen_random_uuid(),
description text not null,
embedding vector(384),
event_type text,
severity integer,
market_impact_data jsonb,
event_timestamp timestamptz,
outcome_accuracy float,
created_at timestamptz default now()
);

create table if not exists trades (
id uuid primary key default gen_random_uuid(),
signal_id text,
instrument text,
direction text,
entry_price float,
exit_price float,
pnl float,
status text,
opened_at timestamptz,
closed_at timestamptz
);

create table if not exists incoming_events (
id uuid primary key default gen_random_uuid(),
raw_text text,
headline text,
source text,
sentiment_score float,
severity_score integer,
processed boolean default false,
created_at timestamptz default now()
);