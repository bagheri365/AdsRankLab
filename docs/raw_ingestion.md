# Original iPinYou raw-log ingestion

AdsRankLab's primary data source is the original iPinYou contest release.

The initial ingestion contract deliberately supports **Seasons 2 and 3**. Those
seasons contain the user-tag field and align with the project's preferred
advertiser/context analysis. Season 1 is not silently forced through the same
schema because its format differs.

## Original event streams

The training release stores separate bzip2-compressed, tab-separated streams:

- `bid.YYYYMMDD.txt.bz2` — bidding logs;
- `imp.YYYYMMDD.txt.bz2` — impressions;
- `clk.YYYYMMDD.txt.bz2` — clicks;
- `conv.YYYYMMDD.txt.bz2` — conversions.

The repository reads `.bz2` files directly; manual decompression is unnecessary.

## Season 2/3 bid schema

Bid logs contain 21 fields:

```text
bid_id
timestamp
user_id
user_agent
ip
region
city
ad_exchange
domain
url
url_id
slot_id
slot_width
slot_height
slot_visibility
slot_format
slot_price
creative_id
bid_price
advertiser_id
user_tags
```

## Season 2/3 impression/click/conversion schema

Outcome-event logs contain 24 fields:

```text
bid_id
timestamp
log_type
user_id
user_agent
ip
region
city
ad_exchange
domain
url
url_id
slot_id
slot_width
slot_height
slot_visibility
slot_format
slot_price
creative_id
bid_price
paying_price
key_page
advertiser_id
user_tags
```

The observed `log_type` is checked against the event stream:

- impression: `1`;
- click: `2`;
- conversion: `3`.

## Evidence boundary

These files are separate event streams. AdsRankLab does **not** infer that every
bid request produced an impression, click, or conversion. Later experiment code
must join events by `bid_id` using an explicitly documented protocol.

Likewise, `paying_price` is observed on outcome-event records, not on every bid
request. Missing counterfactual market prices for lost auctions must not be
filled in as if they were observed.

## Example

```python
from ads_rank_lab.data.ipinyou import read_training_day

frame = read_training_day(
    "data/raw/ipinyou.contest.dataset",
    season=2,
    event_type="imp",
    date="20130606",
    nrows=1000,
)
```
