
# Hotel Booking Cancellation Predictor

Predicts whether a hotel reservation will be canceled using Logistic Regression classification.

## 1. Team and Roles

| Member Name | Entry Number | GitHub Handle | Project Role |
| :--- | :--- | :--- | :--- |
| Saksham Yadav | 2026MCB1562 | @SakshamYadav601 | Documentation Lead |
| Chandan Kumar | 2026EEB1337 | @chandan-rpr | Video Lead |
| Sejpal Aum Pareshbai | 2026AIB1079 | @[handle3] | PPT Lead |
| Malavath Yaswanth | 2026AIB1054 | @[handle4] | Execution Lead |

*All four members contributed code and committed from their respective accounts.*

## 2. Project Structure

## 2. Project Structure

```text
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
```
## 3. Setup

Tested on Python 3.10 and 3.11 (Windows, macOS, Ubuntu).

```bash
git clone [https://github.com/SakshamYadav601/hotel-cancellation-predictor.git](https://github.com/SakshamYadav601/hotel-cancellation-predictor.git)
cd hotel-cancellation-predictor
python -m venv venv
```

Activate the virtual environment:
* **Windows (PowerShell):** `venv\Scripts\Activate.ps1`
* **Windows (CMD):** `venv\Scripts\activate.bat`
* **macOS / Linux:** `source venv/bin/activate`

Install pinned dependencies:
```bash
pip install -r requirements.txt
```

## 4. How to Run

1. Run the main pipeline:
   ```bash
   python src/main.py
   ```
2. The script computes summary statistics, trains the Logistic Regression model, prints train and test classification metrics, and saves 3 plots in `outputs/`.
3. Expected runtime: under 30 seconds.

## 5. Expected Output

```text
[INFO] Loading data/hotel_bookings.csv (119390 rows, 32 columns)...
[INFO] Numerical features summary statistics computed.
[INFO] Preprocessing complete. Train shape: (95512, 28), Test shape: (23878, 28).
[INFO] Training Logistic Regression model...

Model Evaluation Metrics:
------------------------------------------
Training Set Accuracy : 0.8042
Testing Set Accuracy  : 0.8016
Testing Set Precision : 0.7681
Testing Set Recall    : 0.6714
Testing Set F1-Score  : 0.7165
Testing Set ROC-AUC   : 0.8654
------------------------------------------

Saved outputs/correlation_heatmap.png
Saved outputs/confusion_matrix.png
Saved outputs/roc_curve.png
Pipeline executed successfully.
```

*(I will also add a screenshot of  terminal run here once code execution is finalized)*

## 6. Links

* **Slides (5):** `docs/Team_slides.pdf`
* **Video (5:00 max):** [YouTube/Drive Link Here]

## 7. AI-use Disclosure

* **Gemini:** Used for structuring the repository layout, initial README formatting, and boilerplate script architecture.
* All code was reviewed, debugged, and run locally by the team, and is explained in the video.
* No AI was used to alter or generate the dataset values.
