# Insurance Premium Prediction System

## Overview
The Insurance Premium Prediction System is designed to estimate the premium rates for insurance policies based on various features of the customer and policy. This system uses advanced machine learning algorithms to analyze data and predict insurance premiums accurately.

## Features
- Predicts insurance premium rates based on customer and policy features.
- User-friendly interface for input data.
- API endpoints for integration with other applications.
- Configurable model parameters to customize predictions.

## Project Structure
```
mid_sem/
├── api/                # Contains API endpoints
├── models/             # Machine learning models and training scripts
├── config/             # Configuration files
├── tests/              # Test cases for the application
├── requirements.txt    # Project dependencies
└── main.py             # Entry point for the application
```

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/rahulsiddharthdacha/mid_sem.git
   cd mid_sem
   ```  
2. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```  
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide
1. Start the application:
   ```bash
   python main.py
   ```
2. Access the application in your web browser at `http://localhost:5000`.

## API Endpoints
- `POST /predict`
  - Description: Predicts insurance premium.
  - Request Body: 
    ```json
    {
      "age": 30,
      "driving_experience": 10,
      "vehicle_value": 20000,
      ...
    }
    ```
- `GET /status`
  - Description: Returns the status of the API.

## Configuration
The project configuration is managed through files in the `config/` directory. You can adjust settings such as API keys, model parameters, and database configurations.

## Best Practices
- Regularly update your dependencies.
- Validate inputs to the API to prevent errors.
- Create tests to ensure your predictions are accurate and reliable.
- Document any changes to the API endpoints and usage guidelines.

---

This README provides all necessary information to understand and utilize the Insurance Premium Prediction System effectively.