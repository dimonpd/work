import streamlit as st
import joblib
import streamlit.components.v1 as components

rf = joblib.load('rf_model.joblib')
# Создайте форму для ввода данных
st.title('Прогнозирование')
st.write('Введите данные для прогнозирования')

# Создайте поля для ввода данных
X_input = st.number_input('X', min_value=0, max_value=100)

# Создайте кнопку для запуска прогнозирования
if st.button('Прогнозировать'):
    # Прогнозируйте значение
    y_pred = rf.predict([[X_input]])
    st.write(f'Прогнозируемое значение: {y_pred[0]}')