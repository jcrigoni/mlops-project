from app import model_pred

new_data = {'credit_lines_outstanding': 5,
            'loan_amt_outstanding': 5,
            'total_debt_outstanding': 6,
            'income': 0,
            'years_employed': 1,
            'fico_score': 420,
            }


def test_predict():
    prediction = model_pred(new_data)
    assert prediction == 1, "incorrect prediction"
