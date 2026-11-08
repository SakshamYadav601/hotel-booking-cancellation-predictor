
# Hotel Booking Cancellation Predictor

Predicts whether a hotel reservation will be canceled using Logistic Regression classification.

## 1. Team and Roles

| Member Name | Entry Number | GitHub Handle | Project Role |
| :--- | :--- | :--- | :--- |
| Saksham Yadav | 2026MCB1562 | @[your-handle] | Documentation Lead |
| [Teammate 2] | [Entry No.] | @[handle2] | Run & Reproducibility Lead |
| [Teammate 3] | [Entry No.] | @[handle3] | PPT Lead |
| [Teammate 4] | [Entry No.] | @[handle4] | Video Lead |

*All four members contributed code and committed from their respective accounts.*

## 2. Project Structure

hotel-cancellation-predictor/
├── data/
│   └── hotel_bookings.csv
├── docs/
│   └── Team_slides.pdf
├── outputs/
│   ├── correlation_heatmap.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── src/
│   ├── data_prep.py
│   ├── model.py
│   ├── plots.py
│   └── main.py
├── requirements.txt
└── README.md

## 3. Dataset Overview
* Source: Kaggle (jessemostipak/hotel-booking-demand)
* Rows: 119,390 | Columns: 32
* Target: is_canceled (0 = Stayed / Not Canceled, 1 = Canceled)
* Key Features: lead_time, stays_in_week_nights, deposit_type, previous_cancellations, adr

## 4. Setup and Run
```bash
git clone [https://github.com/](https://github.com/)[your-username]/hotel-cancellation-predictor.git
cd hotel-cancellation-predictor
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python src/main.py