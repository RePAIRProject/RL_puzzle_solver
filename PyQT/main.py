from PyQt5 import QtWidgets # import PyQt5 widgets
import sys

# Create the application object
app = QtWidgets.QApplication(sys.argv)

# Create the form object
first_window = QtWidgets.QWidget()

# Set window size
first_window.resize(400, 300)

# Set the form title
first_window.setWindowTitle("The first pyqt program")

# Show form
first_window.show()

# Run the program
sys.exit(app.exec())