# Dataset notes

## Intended dataset

Primary dataset: iPinYou Real-Time Bidding logs.

## Evidence supported by the logs

The project is intended to study:

- click-through-rate prediction;
- probability calibration;
- auction and request context;
- bid/value-aware decision policies;
- cost and win analysis;
- advertiser and traffic-segment analysis.

Conversion modeling may be possible but should be treated cautiously because conversions are sparse.

## Known limitations

The logs should not be presented as a complete platform-side candidate-ranking dataset. In particular, they do not expose all information needed to reconstruct a modern commercial ad auction, including:

- the complete competing candidate slate for every opportunity;
- all competitors' bids;
- proprietary ad-quality signals;
- landing-page quality;
- true advertiser conversion value;
- every modern policy constraint.

All future experiments should distinguish directly observed quantities from model-based or simulated counterfactual quantities.
