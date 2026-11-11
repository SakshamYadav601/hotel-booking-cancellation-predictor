
# Hotel Booking Cancellation Predictor

Predicts whether a hotel reservation will be canceled using Logistic Regression classification.

## 1. Team and Roles

| Member Name | Entry Number | GitHub Handle | Project Role |
| :--- | :--- | :--- | :--- |
| Saksham Yadav | 2026MCB1562 | @SakshamYadav601 | Documentation Lead |
| Chandan Kumar | 2026EEB1337 | @chandan-rpr | Video Lead |
| Sejpal Aum Pareshbai | 2026AIB1079 | @Aum-Sejpal-iitrpr | PPT Lead |
| Malavath Yaswanth | 2026AIB1054 | @Myasvanth2008 | Execution Lead |

*All four members contributed code and committed from their respective accounts.*



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

### Environment & Compatibility
- **Primary Tested Environment:** Windows 11 (Python 3.10+)
- **Cross-Platform Support:** The pipeline relies entirely on standard cross-platform libraries (`pathlib`, `scikit-learn`, `pandas`). Virtual environment commands for both Windows (PowerShell/Bash) and Unix-based systems (macOS/Linux) are provided in the setup instructions.

```bash
git clone https://github.com/SakshamYadav601/hotel-booking-cancellation-predictor
cd hotel-booking-cancellation-predictor
python -m venv venv
```

Activate the virtual environment:

Windows (PowerShell): venv\Scripts\Activate.ps1

Windows (CMD): venv\Scripts\activate.bat

Windows (Git Bash): source venv/Scripts/activate

macOS / Linux: source venv/bin/activate

Install pinned dependencies:
```bash
pip install -r requirements.txt
```

## 4. How to Run

1. Run the main pipeline:
   ```bash
   python -m src.main
   ```
2. The script computes summary statistics, trains the Logistic Regression model, prints train and test classification metrics, and saves 3 plots in `outputs/`.
3. Expected runtime: under 30 seconds.

## 5. Expected Output


```text
=== STARTING HOTEL CANCELLATION PIPELINE ===
[INFO] Loading data/hotel_bookings.csv...
[INFO] Preprocessing complete.
[INFO] Training set shape: (69916, 247)
[INFO] Test set shape:     (17480, 247)
[INFO] Training Logistic Regression model...
[INFO] Model training completed successfully.

[INFO] Evaluating model performance...

Model Evaluation Metrics:
--------------------------------------------------
Training Set Accuracy : 0.7984
Testing Set Accuracy  : 0.7943
Testing Set Precision : 0.6756
Testing Set Recall    : 0.4845
Testing Set F1-Score  : 0.5643
Testing Set ROC-AUC   : 0.8439
--------------------------------------------------

Saved outputs/correlation_heatmap.png
Saved outputs/confusion_matrix.png
Saved outputs/roc_curve.png
Pipeline executed successfully.
```

<img width="420" height="553" alt="image" src="https://github.com/user-attachments/assets/14219b2f-0a11-4584-8428-65cb02ce3346" />


## 6. Links

* **Slides (5):** `docs/Team_slides.pdf`
* **Video (5:00 max):** [YouTube/Drive Link Here]

## 7. AI-use Disclosure

* **Gemini:** Used for structuring the repository layout, initial README formatting, and boilerplate script architecture.
* All code was reviewed, debugged, and run locally by the team, and is explained in the video.
* No AI was used to alter or generate the dataset values.
