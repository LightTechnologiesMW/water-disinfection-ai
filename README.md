# Light Technologies — Adaptive Water Treatment System

**Controlled-release chlorinated resin with AI-optimised disinfection and community feedback for rural Malawi**

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data License: CC-BY 4.0](https://img.shields.io/badge/Data-CC--BY%204.0-green.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Status: Active Pilot](https://img.shields.io/badge/Status-Active%20Pilot-teal.svg)]()

---

## Overview

In Malawi, 5.8 million people lack access to safe drinking water. Climate-driven floods and droughts increasingly contaminate water sources, raising the risk of cholera and diarrhoeal disease — leading killers of children under five. Conventional point-of-use disinfection systems fail rural communities because static chlorine dosing cannot adapt to the variable water chemistry caused by changing environmental conditions.

Light Technologies has developed a **controlled-release chlorinated resin** embedded in a low-cost household device that passively treats water without electricity or trained operators. An **AI/ML system** trained on real-world IoT sensor data dynamically predicts and optimises disinfection performance across flood-affected and drought-concentrated water sources. An **SMS/USSD feedback layer** allows community members to report device performance and water quality concerns from basic feature phones — no internet or smartphone required.

This repository contains hardware documentation, pilot sensor data, and model development code for the system. A full open-source release is a formal deliverable of the current funding phase (target: Months 9–12).

---

## Key results to date

| Metric | Result |
|---|---|
| E. coli reduction | >95% under controlled conditions |
| WHO-compliant Cl₂ residual | 0.2–0.5 mg/L consistently achieved |
| Field units deployed | 5,000+ (NCST-funded pilot, Malawi) |
| Sensor observations (live) | Ongoing — target 100,000+ readings |
| Willingness to adopt at $5–10/unit | 75% (6,000-household survey, Malawi & Rwanda) |

The project was shortlisted for the **Africa Prize for Engineering Innovation** by the Royal Academy of Engineering.

---

## How it works

```
Water source (variable quality)
        │
        ▼
┌─────────────────────────┐
│  Chlorinated resin      │  ← Passive, no electricity
│  household device       │    WHO-compliant Cl₂ dosing
└────────────┬────────────┘
             │
     ┌───────┴────────┐
     ▼                ▼
IoT sensors      SMS / USSD
pH, turbidity,   Community reports
temp, flow rate  via feature phone
     │                │
     └───────┬─────────┘
             ▼
     ┌───────────────┐
     │  AI / ML      │  ← PINN + LSTM + Bayesian optimisation
     │  engine       │    TensorFlow Lite for edge inference
     └───────┬───────┘
             │ Adaptive dosing feedback
             ▼
      Safe drinking water
```

### AI/ML architecture

- **Physics-informed neural networks (PINNs)** — embed chlorination reaction chemistry as model constraints, improving generalisability across variable water conditions
- **LSTM models** — learn temporal patterns from continuous IoT sensor streams, capturing how pH, turbidity, temperature, and flow rate interact to affect chlorine residuals over time
- **Bayesian optimisation** — guides resin formulation experiments, reducing required laboratory trials
- **TensorFlow Lite** — enables edge inference on field devices without internet connectivity

### SMS/USSD feedback layer

Community members can report device faults, water quality concerns, and refill needs via USSD short codes on basic feature phones. This human-generated data feeds directly into model retraining pipelines alongside sensor data, closing the feedback loop between users and the system.

---

## Repository structure

```
/
├── hardware/             # Device schematics, bill of materials, assembly docs
├── data/
│   ├── sample/           # Sample sensor readings (CSV)
│   └── schema/           # Data dictionary and field definitions
├── models/               # Model development notebooks and training scripts
├── ussd/                 # SMS/USSD integration code and API specs
└── docs/                 # Technical documentation and field protocols
```

> **Note:** This repository contains initial datasets, hardware documentation, and model development work from the active pilot phase. Full open-source release — including the complete sensor dataset, trained models, USSD integration code, and deployment documentation — will be completed during the next development phase under CC-BY 4.0 (data) and MIT (code) licenses.

---



## Partners

| Organisation | Role |
|---|---|
| Malawi University of Science and Technology (MUST) | Primary academic partner — laboratory infrastructure and materials characterisation |
| National Commission for Science and Technology (NCST), Malawi | National funder — $80,000 active pilot grant, ethical oversight |
| University of Rwanda | Co-research institution — AI, environmental engineering, cross-country validation |


---

## Funding

This work is supported by the **National Commission for Science and Technology (NCST), Malawi** (active pilot grant, $80,000) and is seeking further investment through UNICEF's Venture Fund to expand the sensor network, build and validate adaptive AI models, and deploy the community feedback layer at scale.

---

## Citation

If you use data or code from this repository, please cite:

```
Light Technologies (2024). Adaptive water treatment system for rural Malawi:
controlled-release chlorinated resin with AI-optimised disinfection.
GitHub: https://github.com/[your-org]/light-technologies-wt
```



## Contact

**Light Technologies**
For research enquiries, partnership discussions, or data access requests, please open an issue or contact [cyruszulu@yahoo.co.uk].

---

## License

- **Code:** [MIT License](LICENSE)
- **Data:** [Creative Commons Attribution 4.0 International (CC-BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
