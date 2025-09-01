
import os
import sys

from matplotlib import pyplot as plt

from app.core.config import settings
from app.services.biometrics.manager_validity import  BiometricVerificationManager
from app.services.nlp.chatbot_services import HealthcareChatbot
from app.services.data.south_africa_data_services import SouthAfricaDataValidator
from app.ML.MODELS.training.traim_triage import TriagePredictor
import streamlit as st


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="SA Healthcare ID Verification & Triage",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize services
    validator = SouthAfricaDataValidator()
    verifier = BiometricVerificationManager()
    chatbot = HealthcareChatbot()

    triage_predictor = TriagePredictor()
    # Add a button to train the model
    if st.sidebar.button("Train Triage Model"):
        with st.spinner("Training model on healthcare data..."):
            train_score, test_score = triage_predictor.train_model()
            st.sidebar.success(f"Model trained! Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
    # Run the application
    render_ui(validator, verifier, chatbot,triage_predictor)


def render_ui(validator, verifier, chatbot, triage_predictor):
    """Render the main application UI"""
    st.title("🏥 South African Healthcare Identity Verification & Triage System")

    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4 = st.tabs([
        "ID Verification",
        "Triage Assessment",
        "AI Chatbot",
        "Data Insights"
    ])

    # Tab 1: ID Verification
    with tab1:
        render_id_verification_tab(validator, verifier)

    # Tab 2: Triage Assessment
    with tab2:
        render_triage_assessment_tab(triage_predictor)

    # Tab 3: AI Chatbot
    with tab3:
        render_chatbot_tab(chatbot)

    # Tab 4: Data Insights
    with tab4:
        render_data_insights_tab(triage_predictor)


def render_id_verification_tab(validator, verifier):
    """Render the ID verification tab"""
    st.header("South African ID Verification")
    st.markdown("Verify patient identity using ID number and facial recognition")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ID Document Information")
        id_number = st.text_input("South African ID Number", placeholder="Enter 13-digit ID number")

        if id_number:
            is_valid, message = validator.validate_sa_id_number(id_number)
            if is_valid:
                st.success(message)
                id_info = validator.extract_info_from_id(id_number)
                st.write(f"**Extracted Information:**")
                st.write(f"- Birth Date: {id_info['birth_date']}")
                st.write(f"- Age: {id_info['age']} years")
                st.write(f"- Gender: {id_info['gender']}")
            else:
                st.error(message)

    with col2:
        st.subheader("Facial Recognition")
        uploaded_file = st.file_uploader("Upload a clear photo of your face", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            if st.button("Verify Identity"):
                with st.spinner("Verifying face..."):
                    is_verified, confidence = verifier.verify_face(uploaded_file)

                    if is_verified and confidence > 0.8:
                        st.success(f"Identity verified with {confidence:.2%} confidence")
                    else:
                        st.error(
                            f"Identity verification failed (confidence: {confidence:.2%}). Please try again with a clearer photo.")


def render_triage_assessment_tab(triage_predictor):
    """Render the triage assessment tab"""
    st.header("Patient Triage Assessment")
    st.markdown("Enter patient information to determine triage priority")

    with st.form("triage_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patient Demographics")
            age = st.slider("Age", 0, 100, 30)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            province = st.selectbox("Province", [
                "Gauteng", "Western Cape", "KwaZulu-Natal", "Eastern Cape",
                "Limpopo", "Mpumalanga", "North West", "Free State", "Northern Cape"
            ])
            has_medical_aid = st.radio("Medical Aid", ["Yes", "No"])
            if has_medical_aid == "Yes":
                medical_scheme = st.selectbox("Medical Scheme", [
                    "Discovery", "Bonitas", "Momentum", "GEMS", "Fedhealth", "Other"
                ])
            else:
                medical_scheme = "None"

        with col2:
            st.subheader("Clinical Information")
            facility_type = st.selectbox("Facility Type", ["Public", "Private"])
            visit_type = st.selectbox("Visit Type", ["Emergency", "Outpatient", "Follow-up"])
            arrival_via_ambulance = st.radio("Arrived by ambulance", ["Yes", "No"])
            hr_bpm = st.slider("Heart Rate (bpm)", 40, 200, 75)
            temp_c = st.slider("Temperature (°C)", 35.0, 41.0, 37.0)
            resp_rate = st.slider("Respiratory Rate (breaths/min)", 10, 40, 16)
            systolic_bp = st.slider("Systolic BP (mmHg)", 80, 200, 120)
            diastolic_bp = st.slider("Diastolic BP (mmHg)", 50, 130, 80)
            o2_sat = st.slider("Oxygen Saturation (%)", 70, 100, 98)
            pain_score = st.slider("Pain Score (0-10)", 0, 10, 3)
            icd10_code = st.text_input("ICD-10 Code (if known)", placeholder="e.g., J06, I10")

        submitted = st.form_submit_button("Determine Triage Priority")

        if submitted:
            # Prepare patient data
            patient_data = {
                'age': age,
                'gender': gender,
                'province': province,
                'has_medical_aid': 1 if has_medical_aid == "Yes" else 0,
                'medical_scheme': medical_scheme,
                'facility_type': facility_type,
                'visit_type': visit_type,
                'arrival_via_ambulance': 1 if arrival_via_ambulance == "Yes" else 0,
                'hr_bpm': hr_bpm,
                'temp_c': temp_c,
                'resp_rate': resp_rate,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'o2_sat': o2_sat,
                'pain_score': pain_score,
                'icd10_code': icd10_code if icd10_code else "Unknown"
            }

            # Predict triage priority
            priority, scores = triage_predictor.predict_triage(patient_data)

            # Display results
            st.subheader("Triage Result")

            if priority == "Red":
                st.error(f"🚨 Priority: {priority} - Immediate care required")
                st.write(
                    "Patient requires immediate medical attention. Conditions in this category are life-threatening.")
            elif priority == "Yellow":
                st.warning(f"⚠️ Priority: {priority} - Urgent care required")
                st.write(
                    "Patient requires urgent medical attention. Conditions are serious but not immediately life-threatening.")
            else:
                st.success(f"✅ Priority: {priority} - Routine care")
                st.write("Patient requires medical attention but can wait for more critical cases to be seen first.")

            # Show contributing factors
            st.write("**Contributing Factors:**")
            factors = []
            if scores[0] > 0:  # Pain score
                factors.append(f"Pain score ({pain_score}/10)")
            if scores[1] > 0:  # Heart rate
                factors.append(f"Heart rate ({hr_bpm} bpm)")
            if scores[2] > 0:  # Oxygen saturation
                factors.append(f"Oxygen saturation ({o2_sat}%)")

            if factors:
                st.write("• " + "\n• ".join(factors))
            else:
                st.write("No critical factors identified")


def render_chatbot_tab(chatbot):
    """Render the AI chatbot tab"""
    st.header("Healthcare Information Chatbot")
    st.markdown("Ask questions about healthcare services in South Africa")

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about healthcare...")

    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get chatbot response
        with st.spinner("Thinking..."):
            bot_response = chatbot.get_response(user_input)

        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})

        # Rerun to update the chat display
        st.rerun()


def render_data_insights_tab(triage_predictor):
    """Render the data insights tab"""
    st.header("Healthcare Data Insights")
    st.markdown("Analysis of triage data and patient demographics")

    demo_data = triage_predictor.load_demo_data()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Triage Priority Distribution")
        priority_counts = demo_data['triage_priority'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(priority_counts.values, labels=priority_counts.index,
               autopct='%1.1f%%', colors=['green', 'orange', 'red'])
        ax.set_title("Triage Priority Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Vital Signs by Priority")
        selected_vital = st.selectbox("Select vital sign",
                                      ['hr_bpm', 'temp_c', 'resp_rate', 'systolic_bp', 'o2_sat'])

        fig, ax = plt.subplots()
        for priority in ['Green', 'Yellow', 'Red']:
            subset = demo_data[demo_data['triage_priority'] == priority]
            ax.hist(subset[selected_vital], alpha=0.7, label=priority)

        ax.set_xlabel(selected_vital)
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.set_title(f"{selected_vital} Distribution by Triage Priority")
        st.pyplot(fig)

    st.subheader("Sample Triage Data")
    st.dataframe(demo_data, use_container_width=True)


if __name__ == "__main__":
    main()