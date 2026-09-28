# iPinYou schema contract

AdsRankLab uses a small canonical schema so downstream experiments do not depend directly on season-specific/raw column spelling.

## Canonical core fields

| Canonical field | Required | Intended role |
|---|---:|---|
| `click` | yes | binary response label |
| `bid_price` | yes | advertiser/DSP bid input |
| `paying_price` | yes | observed clearing/winning cost |
| `conversion` | no | sparse conversion label |
| `floor_price` | no | auction floor/slot price |
| `advertiser_id` | no | advertiser segment |
| `creative_id` | no | ad/creative identifier |
| `user_id` | no | user identifier when present |
| `timestamp` | no | event time |
| `slot_width`, `slot_height` | no | slot geometry |
| `slot_visibility`, `slot_format` | no | slot context |
| `region`, `city` | no | geographic context |
| `user_agent` | no | browser/device context |

The schema layer intentionally allows unknown columns because the exact public release may expose useful features beyond this minimal contract. Unknown columns are reported rather than silently discarded.

## Observed quantities

The project may directly use logged quantities when present, including:

- click labels;
- conversion labels;
- bidding price;
- paying/winning price;
- floor/slot price;
- advertiser and creative identifiers;
- request/user/slot context;
- recorded auction outcomes and cost information.

## Quantities that are not reconstructed as ground truth

AdsRankLab must not silently treat the following as observed when the logs do not provide them:

- complete competing candidate slates;
- every competitor bid;
- proprietary ad-quality scores;
- landing-page quality signals;
- true advertiser conversion value;
- outcomes that would have occurred under an alternative policy;
- clicks/conversions for impressions that were never served.

Any analysis involving these quantities must be labeled as simulated, model-based, or unsupported by the logged evidence.

## Release assumptions

The research plan prefers iPinYou Seasons 2/3 because later releases are expected to provide advertiser information and richer contextual fields useful for segment analysis. Exact raw filenames and field presence should be verified against the locally obtained release before freezing the ingestion pipeline.
