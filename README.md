# Lampedusa Analysis

Analysis of aerosol optical properties and desert dust events using observations from the Lampedusa Climate Observatory and atmospheric model simulations.

## Project Overview

This project focuses on analyzing Aerosol Optical Depth (AOD) and Ångström Exponents (ANG) from the Lampedusa Climate Observatory, a key monitoring station in the Mediterranean Sea managed by ENEA (Italian National Agency for New Technologies, Energy, and Sustainable Economic Development). The observatory is part of NASA's AERONET network and the ACTRIS European Research Infrastructure.

### Key Objectives

1. **Model Validation**: Compare AOD values from the ENEA MINNI-FORAIR-IT air quality model (FARM core) with ground-based observations from Lampedusa, alongside CAMS and HYSPLIT model data.

2. **IMPROVE Formula Calibration**: Test and calibrate the IMPROVE (Interagency Monitoring of Protected Visual Environments) formula for the Mediterranean context, specifically for desert dust events, to improve aerosol-visibility relationships.

3. **AOD Parameterization**: Develop new AOD parameterizations to enhance aerosol forecasting, with the ability to quantify individual aerosol species contributions to atmospheric optical depth.

### Scientific Context

The Mediterranean region is frequently affected by Saharan dust outflows and is highly sensitive to climate change impacts. Lampedusa Island's location, distant from major anthropogenic sources, makes it an ideal site for studying natural aerosol events. This research contributes to:

- Improved air quality forecasting systems
- Better understanding of aerosol radiative effects
- Separation of natural vs. anthropogenic aerosol contributions
- Support for air quality policy development

## Project Structure

```
lampedusa_analysis/
├── config.py              # Configuration settings and parameters
├── main.py               # Main execution script
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── LICENSE              # Project license
├── src/                 # Source code modules
│   ├── data_processing/ # Data loading and preprocessing
│   ├── models/          # Model comparison and validation
│   ├── analysis/        # Statistical analysis and calibration
│   └── visualization/   # Plotting and figure generation
├── notebooks/           # Jupyter notebooks for exploratory analysis
├── tests/               # Unit tests
├── data/                # Data files (not tracked in Git)
│   ├── observations/    # Lampedusa AERONET data
│   ├── model_outputs/   # FARM, CAMS, HYSPLIT outputs
│   └── meteorology/     # Meteorological data
├── outputs/             # Generated results (not tracked in Git)
│   ├── figures/         # Plots and visualizations
│   └── tables/          # Results tables and statistics
└── venv/                # Virtual environment (not tracked in Git)
```

## Setup

### Prerequisites

- Python 3.8 or higher
- WSL/Linux environment (tested on Debian)
- Git with SSH configured for GitHub

### Installation

1. Clone the repository:
```bash
git clone git@github.com:andtoro/lampedusa_analysis.git
cd lampedusa_analysis
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure settings:
```bash
# Edit config.py with your specific paths and parameters
nano config.py
```

## Usage

### Running the Main Analysis

```bash
python main.py
```

### Using Jupyter Notebooks

```bash
jupyter notebook
# Navigate to the notebooks/ directory
```

### Configuration

Edit `config.py` to set:
- Data file paths
- Analysis parameters
- Model configuration
- Output preferences

## Data

Due to file size limitations, data files are not included in this repository. The analysis uses:

- **AERONET observations**: Aerosol optical properties from Lampedusa station
- **FARM model outputs**: Air quality model simulations over Italy
- **CAMS data**: Copernicus Atmosphere Monitoring Service reanalysis
- **HYSPLIT trajectories**: Back-trajectory analysis for dust source identification
- **Meteorological data**: Supporting atmospheric variables

Contact the repository owner for data access information.

## Key Features

- Comparison of ground-based observations with multiple atmospheric models
- Desert dust event detection and analysis
- IMPROVE formula calibration for Mediterranean conditions
- Aerosol species apportionment to AOD
- Statistical validation metrics and visualization tools

## Methodology

1. **Data Processing**: Quality control and harmonization of observational and model data
2. **Event Selection**: Identification of desert dust events using AOD and ANG thresholds
3. **Model Comparison**: Statistical evaluation of FARM, CAMS, and observation agreement
4. **Calibration**: Optimization of IMPROVE coefficients for Mediterranean aerosols
5. **Validation**: Cross-validation and uncertainty quantification

## Results and Outputs

Results are saved in the `outputs/` directory:
- Statistical comparison metrics
- Time series and scatter plots
- Calibrated IMPROVE coefficients
- Aerosol species contribution analysis

## Contributing

This is a research project. For questions or collaboration opportunities, please open an issue or contact the repository maintainer.

## Applications

The methodologies developed in this project support:
- **ENEA operational forecasting**: Daily 3-day air quality forecasts for Italy
- **Copernicus Services**: European-level ensemble forecasting
- **Climate research**: Understanding aerosol-climate interactions in the Mediterranean
- **Policy support**: Informing air quality management strategies

## References

- **Master Thesis**: [Analysis of aerosol optical properties at Lampedusa Climate Observatory](https://etd.adm.unipi.it/theses/available/etd-03092025-162638/)
- AERONET: https://aeronet.gsfc.nasa.gov/
- ACTRIS: https://www.actris.eu/
- ENEA: https://www.enea.it/

## License

See LICENSE file for details.

## Contact

For questions about this research, please open an issue on GitHub.
