# Changelog - ML Heating Underfloor

## [0.2.0] - 2025-04-08

### Added
- Initial release of ML Heating Underfloor addon
- Physics-based machine learning heating control optimized for underfloor heating
- Underfloor-specific thermal defaults (lower outlet temps, higher effectiveness)
- Complete parameter sync with .env configuration
- Cooling mode support with underfloor-specific bounds
- Heat source channel architecture for isolated learning
- Indoor trend protection to prevent parameter drift
- Full InfluxDB v2 integration with features bucket
- Solar correction and PV forecast integration
- Delta forecast calibration for local weather offsets

### Optimized for Underfloor
- CLAMP_MAX_ABS set to 35°C (protects floor covering)
- OUTLET_EFFECTIVENESS at 0.93 (large radiating surface)
- Conservative learning rates for slow thermal mass
- Extended training lookback (1800 hours) for screed slab dynamics
- Slab time constant parameter for Estrich thermal modeling
