
import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QComboBox,
    QPushButton, QTableWidget, QTableWidgetItem, QHBoxLayout, QLineEdit, QFormLayout
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import requests
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import openpyxl

class CryptoPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cryptocurrency Price Predictor")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()

    def initUI(self):
        # Main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()

        # Title label
        self.title_label = QLabel("Cryptocurrency Price Predictor", self)
        self.title_label.setFont(QFont("Arial", 20, QFont.Bold))
        self.title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title_label)

        # Form layout for inputs
        self.form_layout = QFormLayout()
        
        # Cryptocurrency dropdown
        self.crypto_dropdown = QComboBox()
        self.load_cryptocurrencies()
        self.form_layout.addRow("Cryptocurrency:", self.crypto_dropdown)

        # Time period dropdown
        self.time_period_dropdown = QComboBox()
        self.time_period_dropdown.addItems(["24 hours", "7 days", "12 months"])
        self.form_layout.addRow("Time Period:", self.time_period_dropdown)

        # Algorithm dropdown
        self.algorithm_dropdown = QComboBox()
        self.algorithm_dropdown.addItems(["Linear Regression", "ARIMA"])
        self.form_layout.addRow("Algorithm:", self.algorithm_dropdown)

        self.layout.addLayout(self.form_layout)

        # Predict button
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.handle_prediction)
        self.layout.addWidget(self.predict_button)

        # Table for displaying results
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(2)
        self.result_table.setHorizontalHeaderLabels(["Date", "Predicted Price (USD)"])
        self.layout.addWidget(self.result_table)

        # Matplotlib canvas for visualizing predictions
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)

        self.main_widget.setLayout(self.layout)

    def load_cryptocurrencies(self):
        # Fetch top 50 cryptocurrencies from API
        response = requests.get("https://min-api.cryptocompare.com/data/top/mktcapfull?limit=50&tsym=USD")
        if response.ok:
            self.cryptocurrencies = [c["CoinInfo"]["Name"] for c in response.json()["Data"]]
            self.crypto_dropdown.addItems(self.cryptocurrencies)

    def handle_prediction(self):
        selected_crypto = self.crypto_dropdown.currentText()
        time_period = self.time_period_dropdown.currentText()
        algorithm = self.algorithm_dropdown.currentText()

        # Define time parameters
        time_params = {"24 hours": 24, "7 days": 168, "12 months": 365}
        limit = time_params[time_period]
        aggregate = 1

        # Fetch historical data
        response = requests.get(
            "https://min-api.cryptocompare.com/data/v2/histohour",
            params={"fsym": selected_crypto, "tsym": "USD", "limit": limit, "aggregate": aggregate},
        )

        if response.ok:
            data = response.json()["Data"]["Data"]
            timestamps = [datetime.fromtimestamp(d["time"]) for d in data]
            prices = [d["close"] for d in data]

            # Predict prices
            if algorithm == "Linear Regression":
                model = LinearRegression()
                x = [[t.timestamp()] for t in timestamps]
                model.fit(x, prices)
                predictions = model.predict(x)

            elif algorithm == "ARIMA":
                model = ARIMA(prices, order=(5, 1, 0))
                fit_model = model.fit()
                predictions = fit_model.predict(start=1, end=len(prices), dynamic=False)

            self.update_table(timestamps, predictions)
            self.update_chart(timestamps, prices, predictions)

    def update_table(self, timestamps, predictions):
        self.result_table.setRowCount(len(timestamps))
        for i, (timestamp, prediction) in enumerate(zip(timestamps, predictions)):
            self.result_table.setItem(i, 0, QTableWidgetItem(timestamp.strftime('%Y-%m-%d %H:%M:%S')))
            self.result_table.setItem(i, 1, QTableWidgetItem(f"{prediction:.2f}"))

    def update_chart(self, timestamps, prices, predictions):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Plot historical prices
        ax.plot(timestamps, prices, label="Historical Prices", color="blue")
        
        # Plot predicted prices
        ax.plot(timestamps, predictions, label="Predicted Prices", color="red", linestyle="--")

        # Formatting the chart
        ax.set_title("Cryptocurrency Price Prediction")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (USD)")
        ax.legend()
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CryptoPredictorApp()
    window.show()
    sys.exit(app.exec_())
