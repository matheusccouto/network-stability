import streamlit as st
import network_stability

st.title('Network Stability')

hours = st.text_input(label='Hours', value=0)
minutes = st.text_input(label='Minutes', value=0)
seconds = st.text_input(label='Seconds', value=0)

speed_button = st.button(label='Run Speed Test')
connection_button = st.button(label='Run Connection Test')

if speed_button:
    with st.spinner('Wait for it...'):
        net = network_stability.NetworkTest()
        net.speed_test_interval(hours=int(hours), minutes=int(minutes), seconds=int(seconds))
    st.success('Speed test completed.')
    fig = net.report_speed('speed.png')
    st.pyplot(fig, bbox_inches='tight')

if connection_button:
    with st.spinner('Wait for it...'):
        net = network_stability.NetworkTest()
        net.connection_test_interval(hours=int(hours), minutes=int(minutes), seconds=int(seconds))
    st.success('Connection test completed.')
    fig = net.report_connection('connection.png')
    st.pyplot(fig, bbox_inches='tight')