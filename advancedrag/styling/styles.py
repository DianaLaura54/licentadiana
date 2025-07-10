import base64


def get_base64_image(image_path):
    """Convert image to base64 string for CSS embedding"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def get_css():
    # Path to your bot icon
    bot_icon_path = r"E:\AN 4\licenta\advancedrag\icons\bot_icon.png"

    # Get base64 image
    base64_image = get_base64_image(bot_icon_path)

    # Determine bot message CSS based on whether image loaded successfully
    if base64_image:
        bot_message_css = f"""
        /* Enhanced Bot Message (Answer) Display with Custom Icon */
        .bot-message {{
            background: linear-gradient(135deg, #ff3366, #e6004c);
            color: #ffb3d1;
            padding: 20px 20px 20px 85px; /* Increased left padding for icon space */
            margin: 15px 0;
            border-radius: 20px 20px 20px 5px;
            border: none;
            box-shadow: 0 6px 20px rgba(255, 51, 102, 0.3);
            font-size: 15px;
            line-height: 1.6;
            position: relative;
            max-width: 85%;
            margin-right: auto;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }}

        .bot-message::before {{
            content: "";
            position: absolute;
            left: 15px;
            top: 20px;
            width: 45px;
            height: 45px;
            background-image: url('data:image/png;base64,{base64_image}');
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            border-radius: 50%;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
            border: 2px solid #ffd1dc;
        }}

        .bot-message::after {{
            content: "A: ";
            position: absolute;
            left: 65px;
            top: 22px;
            font-weight: bold;
            color: #ffd1dc;
            font-size: 18px;
        }}
        """
    else:
        # Fallback if image doesn't load
        bot_message_css = """
        /* Enhanced Bot Message (Answer) Display - Fallback */
        .bot-message {
            background: linear-gradient(135deg, #ff3366, #e6004c);
            color: #ffb3d1;
            padding: 20px;
            margin: 15px 0;
            border-radius: 20px 20px 20px 5px;
            border: none;
            box-shadow: 0 6px 20px rgba(255, 51, 102, 0.3);
            font-size: 15px;
            line-height: 1.6;
            position: relative;
            max-width: 85%;
            margin-right: auto;
            text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
        }

        .bot-message::before {
            content: "🤖 A: ";
            font-weight: bold;
            color: #ffd1dc;
            margin-right: 8px;
            font-size: 18px;
        }
        """

    return f"""
    <style>
    /* Main background - Soft pink gradient */
    .stApp {{
        background: linear-gradient(135deg, #ffe6f2 0%, #ffcce6 100%);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #ffb6c1;  /* Light pink text for entire app */
    }}

    /* Make ALL headers and accent text light pink */
    div.stRadio > label > div:first-child,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stText, .stTextInput, .stSelectbox,
    .yellow-text {{
        color: #ffb6c1;  /* Light pink */
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.1);
    }}

    /* Background animation */
    @keyframes gradientBG {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}

    /* Red-pink header */
    .main-header {{
        text-align: center;
        background: linear-gradient(90deg, #ff3366, #cc0033);
        color: #ffd1dc;  /* Light pink text on header */
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 6px 12px rgba(204, 0, 51, 0.2);
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);
    }}

    /* Red-pink buttons with light pink text */
    .stButton>button {{
        background: linear-gradient(45deg, #ff3366, #e6004c);
        color: #ffd1dc;  /* Light pink text */
        border-radius: 25px;
        border: none;
        padding: 10px 20px;
        font-size: 14px;
        transition: all 0.3s ease;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(255, 51, 102, 0.3);
    }}

    .stButton>button:hover {{
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(255, 51, 102, 0.4);
    }}

    /* Radio button options */
    div.stRadio > div[role="radiogroup"] > label {{
        color: #ffb6c1;  /* Light pink */
        font-weight: 500;
    }}

    /* Input fields */
    .stTextInput>div>div>input {{
        border: 1px solid #ff99c2;
        border-radius: 10px;
        color: #ff80ab;  /* Medium light pink */
        background: rgba(255, 255, 255, 0.9);
    }}

    /* Selectbox styling */
    .stSelectbox>div>div>div {{
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid #ff99c2;
        border-radius: 10px;
        color: #cc0033;
    }}

    /* Selectbox dropdown options */
    .stSelectbox>div>div>div>div {{
        color: #cc0033;
        background: rgba(255, 255, 255, 0.95);
    }}

    /* Selectbox dropdown menu */
    div[data-baseweb="select"] > div {{
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid #ff99c2;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(255, 51, 102, 0.2);
    }}

    /* Individual dropdown options */
    div[role="option"] {{
        color: #cc0033;
        background: transparent;
        padding: 8px 12px;
    }}

    /* Hover state for dropdown options */
    div[role="option"]:hover {{
        background: rgba(255, 153, 194, 0.2);
        color: #990026;
    }}

    /* Selected option in dropdown */
    div[role="option"][aria-selected="true"] {{
        background: rgba(255, 51, 102, 0.1);
        color: #cc0033;
        font-weight: 600;
    }}

    /* Enhanced Chat Container */
    .chat-container {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 182, 193, 0.3);
        box-shadow: 0 8px 32px rgba(255, 51, 102, 0.1);
    }}

    /* Enhanced User Message (Question) Display */
    .user-message {{
        background: linear-gradient(135deg, #ff3366, #e6004c);
        color: #ffffff;
        padding: 15px 20px;
        margin: 15px 0;
        border-radius: 20px 20px 5px 20px;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 51, 102, 0.3);
        font-size: 16px;
        font-weight: 500;
        position: relative;
        max-width: 80%;
        margin-left: auto;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }}

    .user-message::before {{
        content: "Q: ";
        font-weight: bold;
        color: #ffd1dc;
        margin-right: 8px;
    }}

    {bot_message_css}

    /* Chat Input Container */
    .chat-input-container {{
        background: rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 182, 193, 0.3);
        box-shadow: 0 4px 15px rgba(255, 51, 102, 0.1);
    }}

    /* Send Button Enhancement */
    .send-button button {{
        background: linear-gradient(45deg, #ff3366, #cc0033);
        color: #ffffff;
        border-radius: 15px;
        padding: 12px 25px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 12px rgba(255, 51, 102, 0.4);
        transition: all 0.3s ease;
    }}

    .send-button button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 51, 102, 0.5);
    }}

    /* Generate Button Enhancement */
    .generate-button button {{
        background: linear-gradient(45deg, #ff99c2, #ff66a3);
        color: #ffffff;
        border-radius: 20px;
        padding: 12px 20px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 12px rgba(255, 153, 194, 0.3);
    }}

    /* Empty Chat State */
    .empty-chat {{
        text-align: center;
        color: #ff99c2;
        font-size: 18px;
        font-style: italic;
        padding: 40px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        border: 2px dashed #ff99c2;
    }}

    /* Score displays and other text elements */
    .score-display, .chat-header, .stMarkdown p {{
        color: #ffb6c1;  /* Light pink */
    }}

    /* Message animations */
    .user-message, .bot-message {{
        animation: slideIn 0.3s ease-out;
    }}

    @keyframes slideIn {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    /* TTS Button styling */
    .stButton[data-testid*="tts"] button {{
        background: linear-gradient(45deg, #ff80ab, #ff4081);
        color: white;
        border-radius: 20px;
        padding: 8px 15px;
        font-size: 12px;
        margin-top: 10px;
    }}
    </style>
    """