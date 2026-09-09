import streamlit as st
import pandas as pd
import joblib

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Speech Readiness Classifier", layout="wide")
st.title("🗣️ Public Speaking Readiness Prediction Classifier")
st.markdown("Enter the respondent's information below to generate predictions from all three models.")

# --- 2. LOAD MODELS & PREPROCESSORS (Cached for performance) ---
@st.cache_resource
def load_ml_assets():
    try:
        assets = {}

        assets['lr']  = joblib.load('./models/best_lr.joblib')
        assets['rfc'] = joblib.load('./models/best_rfc.joblib')
        assets['svc'] = joblib.load('./models/best_svc.joblib')

        assets['label_encoders'] = joblib.load('./preprocessers/label_encoder.joblib')

        assets['age_encoder'] = joblib.load('./preprocessers/age_encoder.joblib')
        assets['edu_lvl_encoder'] = joblib.load('./preprocessers/edu_lvl_encoder.joblib')
        assets['edu_status_encoder'] = joblib.load('./preprocessers/edu_status_encoder.joblib')
        assets['exp_lvl_encoder'] = joblib.load('./preprocessers/exp_lvl_encoder.joblib')
        assets['gender_encoder'] = joblib.load('./preprocessers/gender_encoder.joblib')
        assets['opportunity_impact_encoder'] = joblib.load('./preprocessers/opportunity_impact_encoder.joblib')
        assets['self_esteem_impact_encoder'] = joblib.load('./preprocessers/self_esteem_impact_encoder.joblib')
        assets['ph_avoid_encoder'] = joblib.load('./preprocessers/ph_avoid_encoder.joblib')
        assets['physical_symptoms_encoder'] = joblib.load('./preprocessers/physical_symptoms_encoder.joblib')
        assets['profession_encoder']     = joblib.load('./preprocessers/profession_encoder.joblib')

        assets['standard_scaler'] = joblib.load('./preprocessers/standard_scaler.joblib')

        return assets 
    except Exception as e:
        st.warning(f"ML assets not loaded yet. Uncomment the joblib lines when ready. Error: {e}")
        return None

assets = load_ml_assets()

# --- 3. UI FORM ---
with st.form("anxiety_inputs"):
    st.subheader("Demographics & Background")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age_group = st.selectbox("အသက်အရွယ် (Age Group)", 
                                 ['Under 20', '20 - 30', '31 - 40', 'Above 40'])
        current_profession = st.selectbox("လက်ရှိအလုပ်အကိုင် (Current Profession)", 
                                          ['Student (ကျောင်းသား/သူ)', 'Employee / Worker (ဝန်ထမ်း)', 
                                           'Businessman / Entrepreneur (ကိုယ်ပိုင်စီးပွားရေးလုပ်ငန်းရှင်)', 
                                           'Freelancer', 'Other'])
        education_status = st.selectbox("လက်ရှိ တက်ရောက်နေခြင်း ရှိ၊ မရှိ (Education Status)", 
                                        ['Graduated', 'Attending'])
        impact_on_opportunities = st.selectbox("လူရှေ့စကားပြောရမှာ ကြောက်သည့်စိတ်... (Impact on Opportunities)", 
                                               ['Yes (ကြုံဖူးပါသည်)', 'No (မကြုံဖူးပါ)'])
        ph_avoid = st.selectbox("စကားပြော ကြောက်ရွှံ့ခြင်းကြောင့်... (Ways of Avoiding Phone Calls)", 
                                ['No Avoidance', 'Direct Avoidance', 'Excuse / Pretext', 'Delegation', 'Channel Switch'])

    with col2:
        gender = st.selectbox("ကျား/မ (Gender)", 
                              ['Male (ကျား)', 'Female (မ)'])
        education_level = st.selectbox("ပညာရေးအဆင့် (Education Level)", 
                                       ['Student', 'High School', 'Diploma', 'Bachelor', 'Post Graduate'])
        exp_lvl = st.selectbox("လုပ်ငန်းအတွေ့အကြုံ (Experience Level)", 
                               ['Less than 1 year', '1 - 3 years', '4 - 7 years', 'More than 7 years'])
        impact_on_self_esteem = st.selectbox("စကားပြောဆိုဆက်သွယ်မှု အားနည်းချက်... (Impact on Self-Esteem)", 
                                             ['Yes (ကျဆင်းရပါသည်)', 'No (မကျဆင်းပါ)'])
        physical_symptoms = st.selectbox("စကားပြောနေစဉ်အတွင်း စိတ်လှုပ်ရှားမှုကြောင့်... (Physical Symptoms Severity)", 
                                         ['None', 'Mild Symptoms', 'Moderate Symptoms', 'Severe Symptoms', 'Extreme Symptoms'])

    st.markdown("---")
    st.subheader("Behavioral Assessment (1 = Lowest, 5 = Highest)")
    
    col3, col4 = st.columns(2)
    
    with col3:
        fear_judgment = st.slider("ဝေဖန်လှောင်ပြောင်ကြမလားဆိုသည့် စိုးရိမ်စိတ် (Fear of Negative Social Judgment)", 1, 5, 3)
        pref_texting = st.slider("Text Message (စာတို) ပို့၍ ဆက်သွယ်ရခြင်းကို ပိုမိုအားသန်မှု (Preference for Texting)", 1, 5, 3)
        
    with col4:
        avoidance_behavior = st.slider("စကားပြောရမည့် အလှည့်ကို ရှောင်လွှဲလိုစိတ် (Avoidance Behavior of Public Speaking)", 1, 5, 3)
        comfort_social = st.slider("Social Media ပေါ်တွင် မိမိကိုယ်မိမိ ပိုမိုလွတ်လပ်စွာ ဖော်ပြနိုင်သည်ဟု ခံစားရမှု (Comfort Level on Social Media)", 1, 5, 3)

    # Submit button
    submitted = st.form_submit_button("Predict Anxiety Levels", type="primary", use_container_width=True)

# --- 4. PROCESSING & PREDICTION ---
if submitted:
    # Handle the specific logic for 'Post Graduate' -> 'Master'
    mapped_education_level = 'Master' if education_level == 'Post Graduate' else education_level

    # Create the single row dictionary matching your exact dataframe column names
    input_dict = {
        'age_group': [age_group],
        'gender': [gender],
        'current_profession': [current_profession],
        'education_level': [mapped_education_level],
        'education_status': [education_status],
        'exp_lvl': [exp_lvl],
        'impact_on_opprtunities': [impact_on_opportunities],
        'impact_on_self_esteem': [impact_on_self_esteem],
        'ph_avoid': [ph_avoid],
        'physical_symptoms_of_nervousness': [physical_symptoms],
        'fear_of_negative_social_judgement': [fear_judgment],
        'avoidance_behaviour_of_public_speaking': [avoidance_behavior],
        'preference_for_texting': [pref_texting],
        'comfort_lvl_on_social_media_space': [comfort_social]
    }

    # Convert to Dataframe
    input_df = pd.DataFrame(input_dict)
    
    st.success("Data successfully compiled into DataFrame!")
    st.dataframe(input_df) # Visual confirmation for your presentation

    # --- Prediction Execution Block ---
    # Uncomment and adjust this block once your joblib files are linked

    try:
        df = input_df.copy()

        # Preprocess the data

        # Age Group
        df['age_group'] = assets['age_encoder'].transform(df[['age_group']]) + 1

        # Gender
        gender_encoded = assets['gender_encoder'].transform(df[['gender']])
        gender_df = pd.DataFrame(
            gender_encoded,
            columns=assets['gender_encoder'].get_feature_names_out(['gender']),
            index=df.index
        )
        df = pd.concat([df.drop('gender', axis=1), gender_df], axis=1)
        # Multicollinearity
        df.rename(columns={'gender_Male (ကျား)': 'is_male'}, inplace=True)
        df.drop(columns=['gender_Female (မ)'], inplace=True)

        # Current Profession
        profession_encoded = assets['profession_encoder'].transform(df[['current_profession']])
        profession_df = pd.DataFrame(
            profession_encoded,
            columns=assets['profession_encoder'].get_feature_names_out(['current_profession']),
            index=df.index
        )
        profession_df.rename(columns={'current_profession_Student (ကျောင်းသား/သူ)': 'is_student'}, inplace=True)
        profession_df.rename(columns={'current_profession_Employee / Worker (ဝန်ထမ်း)': 'is_employee'}, inplace=True)
        profession_df.rename(columns={'current_profession_Businessman / Entrepreneur (ကိုယ်ပိုင်စီးပွားရေးလုပ်ငန်းရှင်)': 'is_businessman'}, inplace=True)
        profession_df.rename(columns={'current_profession_Freelancer': 'is_freelancer'}, inplace=True)
        df = pd.concat(
            [df.drop(columns=['current_profession']), profession_df],
            axis=1
        )

        # Experience Level
        df['exp_lvl'] = assets['exp_lvl_encoder'].transform(df[['exp_lvl']]) + 1

        # Education Level
        df['education_level'] = assets['edu_lvl_encoder'].transform(df[['education_level']]) + 1
        # Education Status
        df['education_status'] = assets['edu_status_encoder'].transform(df[['education_status']]) + 1

        # Impact on Opportunities
        opportunity_impact_encoded = assets['opportunity_impact_encoder'].transform(df[['impact_on_opprtunities']])
        oppor_impact_df = pd.DataFrame(
            opportunity_impact_encoded,
            columns=[
                'impact_on_opportunities'
            ],
            index=df.index
        )
        df = pd.concat(
            [df.drop(columns=['impact_on_opprtunities']), oppor_impact_df],
            axis=1
        )

        # Impact on Self-Esteem
        self_est_impact_encoded = assets['self_esteem_impact_encoder'].transform(df[['impact_on_self_esteem']])
        impact_on_self_esteem = pd.DataFrame(
            self_est_impact_encoded,
            columns=[
                'impact_on_self_esteem'
            ],
            index=df.index
        )
        df = pd.concat(
            [df.drop(columns=['impact_on_self_esteem']), impact_on_self_esteem],
            axis=1
        )

        # Phone Avoidance
        encoded_array = assets['ph_avoid_encoder'].transform(df[['ph_avoid']])
        feature_names = assets['ph_avoid_encoder'].get_feature_names_out(['ph_avoid'])
        ph_avoid_encoded = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
        ph_avoid_encoded = ph_avoid_encoded.drop(columns=['ph_avoid_Unknown'], errors='ignore')
        rename_dict = {
            'ph_avoid_Channel Switch': 'ph_avoid_channel_switch',
            'ph_avoid_Delegation': 'ph_avoid_delegation',
            'ph_avoid_Direct Avoidance': 'ph_avoid_direct',
            'ph_avoid_Excuse / Pretext': 'ph_avoid_excuse',
            'ph_avoid_No Avoidance': 'ph_avoid_no',
        }
        ph_avoid_encoded.rename(columns=rename_dict, inplace=True)
        df = pd.concat([df.drop(columns=['ph_avoid']), ph_avoid_encoded], axis=1)

        # Physical Symptoms of Nervousness
        df['physical_symptoms_of_nervousness'] = assets['physical_symptoms_encoder'].fit_transform(df[['physical_symptoms_of_nervousness']]) + 1

        # Drop unnecessary columns
        features_to_drop = [
            'ph_avoid_delegation',
            'ph_avoid_channel_switch',
            'is_student',
            'is_employee',
            'ph_avoid_excuse'
        ]
        df = df.drop(columns=features_to_drop)

        # Reorder columns
        ref_columns = [
            'age_group',
            'exp_lvl',
            'physical_symptoms_of_nervousness',
            'fear_of_negative_social_judgement',
            'avoidance_behaviour_of_public_speaking',
            'preference_for_texting',
            'comfort_lvl_on_social_media_space',
            'is_male',
            'is_freelancer',
            'is_businessman',
            'education_level',
            'education_status',
            'impact_on_opportunities',
            'impact_on_self_esteem',
            'ph_avoid_direct',
            'ph_avoid_no'
        ]
        df = df[ref_columns]

        st.success("Dataframe successfully preprocessed!")
        st.dataframe(df)

        # Standard Scaler
        df = assets['standard_scaler'].transform(df)

        # Show Preporcessed DataFrame
        st.dataframe(df)        

        # Make predictions and decode class labels
        label_encoder = assets['label_encoders']
        lr_label = label_encoder.inverse_transform(assets['lr'].predict(df))[0]
        svc_label = label_encoder.inverse_transform(assets['svc'].predict(df))[0]
        rfc_label = label_encoder.inverse_transform(assets['rfc'].predict(df))[0]
        
        # Display side-by-side results
        st.markdown("### Model Predictions")
        st.markdown("#### Your Public Speaking Readiness Level is:")
        res1, res2, res3 = st.columns(3)
        
        res1.metric("Logistic Regression", lr_label)
        res2.metric("Support Vector Classifier", svc_label)
        res3.metric("Random Forest Classifier", rfc_label)
        
    except Exception as e:
        st.error(f"Prediction Error: {e}")
