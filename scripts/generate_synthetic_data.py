import os
import random
import csv
from pathlib import Path

# Setup classes
CLASSES = [
    "Safe",
    "Digital Arrest Scam",
    "OTP Scam",
    "Bank KYC Scam",
    "Investment Scam",
    "Loan Scam",
    "QR Code Scam",
    "Job Scam",
    "Romance Scam",
    "Crypto Scam"
]

# Random parameter pools
GREETINGS = ["Hello", "Hi", "Dear customer", "Hey", "Good morning", "Respected sir/madam", "Hi there"]
SIGNOFFS = [
    "Please reply soon.",
    "Let me know.",
    "Act fast to resolve this.",
    "Thanks.",
    "Regards.",
    "Do not ignore this message.",
    "Have a good day."
]
BANKS = ["SBI", "HDFC", "ICICI", "Axis Bank", "Punjab National Bank", "Kotak Mahindra"]
AMOUNTS = ["10,000", "25,000", "50,000", "1,50,000", "5,00,000", "15,00,000"]
NAMES = ["Sanjay Kumar", "Amit Shah", "Megnand Sharma", "Officer Mishra", "Tina", "David", "Vikram Rathore"]
CRYPTO_COINS = ["Bitcoin", "USDT", "Ethereum", "Doge", "Solana"]

def get_safe_templates() -> list:
    return [
        "Are we still meeting for lunch at {time} today? Let me know.",
        "Hey, I will be a bit late for dinner tonight. Please don't wait for me.",
        "Could you please share the project presentation deck for tomorrow's meeting?",
        "Your package from Amazon is out for delivery and will reach you by {time}.",
        "Hi, I wanted to check if you received my email regarding the job application status?",
        "Can you send me the address of the restaurant we visited last weekend?",
        "Happy birthday! Wishing you a wonderful year ahead filled with joy.",
        "I have completed the code review for the pull request. Let's merge it.",
        "Hey, can you please transfer {amount} rupees to my account for the concert tickets?",
        "Hello, this is to confirm your doctor appointment scheduled for tomorrow at {time}."
    ]

def get_digital_arrest_templates() -> list:
    return [
        "This is {name} calling from the CBI Mumbai headquarters. Your Aadhaar card has been linked to a money laundering syndicate involving {amount} rupees. You are under digital arrest. You must join a WhatsApp video call immediately. Do not talk to anyone or leave your house, or you will face immediate arrest.",
        "I am officer {name} from Crime Branch. An illegal passport containing drugs was caught in Mumbai customs under your Aadhaar details. The Supreme Court has issued a warrant. You are under digital arrest. Keep your phone video on 24/7. Transfer your funds to RBI security account for verification.",
        "Telecom authority TRAI alert. Your mobile number is blacklisted for sending illegal texts. I am connecting you to Mumbai Police Crime Branch. Officer {name} here, you are under digital arrest for financial fraud. Cooperate or go to jail for 7 years.",
        "CBI Notarized Supervision Acknowledgement. You are accused of opening 15 fake bank accounts using your identity documents. You are placed under digital arrest starting now. Do not notify your family. Cooperate on WhatsApp video call to avoid lockup."
    ]

def get_otp_templates() -> list:
    return [
        "Hi, I am from the {courier} delivery service. I am standing near your gate but I need a 6-digit OTP to deliver your package. Please read out the OTP sent to your phone.",
        "Alert: Your transaction of {amount} rupees on Amazon is pending. To cancel this unauthorized transaction, please share the OTP code sent to your phone immediately.",
        "Hi, your electricity connection will be cut tonight at 9:30 PM due to unpaid bills. I can update it in the database. Please tell me the OTP sent to your phone number now.",
        "Dear customer, your credit card point redemption request for {amount} rupees is successful. To credit the cash to your bank, tell me the verification OTP you just received."
    ]

def get_kyc_templates() -> list:
    return [
        "Dear customer, your {bank} bank account has been suspended today. KYC verification has expired. Please click this link to update your KYC documents and restore services: {url_link}",
        "Hello, I am the branch manager of {bank}. Your ATM debit card has been blocked for security. I need your card number, CVV, and net banking password to complete your KYC update.",
        "Alert: Your netbanking account will block permanently within 24 hours. Update PAN card and KYC now. Click here to login to your bank portal: {url_link}",
        "Dear user, your mobile SIM KYC is incomplete and your number will be deactivated by tonight. Please contact our support team and share your Aadhaar details and OTP."
    ]

def get_investment_templates() -> list:
    return [
        "Welcome to the VIP trading group '{group_name}' hosted by {name}. We provide daily stock trading signals and IPO allocations with 300% guaranteed returns. See screenshots of profits of {amount} rupees made by our members. To start trading, deposit money into our institutional account.",
        "Make massive wealth daily with stock market training. We guarantee 50% profits every day on short term options. Join our Telegram vip channel. Transfer {amount} rupees today and trade under supervision of {name}.",
        "Special IPO allocation opportunity! Get guaranteed shares of high demand companies. Limited seats left. Transfer capital to our trading account. Thousands of members are already making money.",
        "Earn passive income from stock investment. We handle the trading for you. Guaranteed returns of 20% weekly. Deposit funds directly to our coordinator to unlock your VIP membership."
    ]

def get_loan_templates() -> list:
    return [
        "Get instant personal loans up to {amount} rupees with zero credit score check and low interest rates of 2%. To disburse the money, you must pay a processing fee of 4,999 rupees immediately.",
        "Congratulations! Your loan application for {amount} rupees is approved. To receive the funds in your account, please pay a refundable security deposit of 9,999 rupees to the bank officer.",
        "Need urgent cash? Instant loan approved within 5 minutes. No paperwork required. Pay initial documentation charge of 2,500 rupees to get the cash. Click this link to apply: {url_link}",
        "Emergency loan offer! Loans up to 10 lakhs. We only require processing charge upfront. Pay now or your loan file will be cancelled and blacklisted from all banks."
    ]

def get_qr_templates() -> list:
    return [
        "I want to purchase your item listed on OLX. I am sending you a QR code on WhatsApp. Just scan the QR code and enter your UPI PIN to receive the payment of {amount} rupees in your account.",
        "Hi, I have sent you a payment request of {amount} rupees on Google Pay. You just need to scan this QR code and approve it to credit the cash directly to your bank account.",
        "Scan this scan card QR code to claim your cashback voucher of {amount} rupees. Once scanned, enter your UPI password and the money will be sent to your account.",
        "Sir, I am paying you advance money. Scanning this QR code will deposit the funds. Do not worry, just scan it, enter your PIN and click receive."
    ]

def get_job_templates() -> list:
    return [
        "Earn {amount} rupees daily by doing part-time work from home. Task is simple: like YouTube videos, rating hotels on Maps, or subscribing to channels. No signup fees. Join our Telegram group now.",
        "Hi, I am HR manager at {company}. We have a vacancy for part-time assistant. Earn {amount} per day. Get paid 100 rupees per video like. Share your UPI and start tasks.",
        "Task completed! You earned 500 rupees. To withdraw this and unlock premium high-paying tasks of Level 2, you must deposit 2,000 rupees. This is fully refundable capital.",
        "Work from home opportunity! Earn massive commissions by rating movies. To continue receiving tasks and withdraw your balance, pay the deposit of 5,000 rupees to our merchant account."
    ]

def get_romance_templates() -> list:
    return [
        "My love, I have shipped a surprise package for you from London. It has a gold chain, a Rolex, and {amount} cash. The customs agent at airport called. You must pay {amount_small} rupees clearance fee to release it.",
        "Dearest, I sent you a gift box containing a new iPhone and designer bags. The courier company says it is held at Delhi airport for customs duty. Please deposit {amount_small} rupees into the customs account so they deliver it.",
        "I am so happy I met you on Tinder. I want to buy a house and settle down with you. I am sending a package with diamonds. The customs officer {name} wants tax clearance fee. Please help me pay it.",
        "Honey, my card is blocked while traveling. Can you please pay {amount_small} rupees to the airport customs authority for my luggage fee? I will return it as soon as I arrive."
    ]

def get_crypto_templates() -> list:
    return [
        "Double your {coin} and {coin} today! Send your crypto to our professional trading smart contract address and receive 200% return back in 24 hours. Safe and guaranteed.",
        "Invest in the presale of the next 100x meme coin! Transfer {coin} or {coin} to our contract address. Get early allocation. The coin list on Binance next week, buy now.",
        "Earn 10% daily ROI on our decentralized crypto staking platform. Create an account, deposit {coin} to the wallet address, and watch your balance grow. Withdraw anytime.",
        "Exclusive crypto arbitrage signals! Sign up on our private exchange and buy {coin} at low price, sell at high price. Deposit at least 500 {coin} to activate the bot."
    ]

def generate_dataset(output_path: Path, samples_per_class: int = 60) -> None:
    """Generates a synthetic scam dataset and writes it to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    random.seed(42)
    dataset = []

    # Map classes to their templates generator
    template_map = {
        "Safe": get_safe_templates,
        "Digital Arrest Scam": get_digital_arrest_templates,
        "OTP Scam": get_otp_templates,
        "Bank KYC Scam": get_kyc_templates,
        "Investment Scam": get_investment_templates,
        "Loan Scam": get_loan_templates,
        "QR Code Scam": get_qr_templates,
        "Job Scam": get_job_templates,
        "Romance Scam": get_romance_templates,
        "Crypto Scam": get_crypto_templates
    }

    for label in CLASSES:
        templates = template_map[label]()
        for i in range(samples_per_class):
            template = random.choice(templates)
            
            # Format parameters dynamically
            formatted_text = template.format(
                time=f"{random.randint(1, 12)}:{random.choice(['00', '15', '30', '45'])} {random.choice(['AM', 'PM'])}",
                amount=random.choice(AMOUNTS),
                amount_small=f"{random.randint(5, 45)}000",
                name=random.choice(NAMES),
                bank=random.choice(BANKS),
                courier=random.choice(["DHL", "BlueDart", "Delhivery", "Post Office"]),
                group_name=random.choice(["D18 Exploring Profit Methods", "VIP Stock Academy", "Wealth Creators Club"]),
                url_link=f"http://{random.choice(['verification', 'login', 'update', 'secure'])}-{random.choice(BANKS).lower().replace(' ', '')}.com",
                company=random.choice(["Google Maps rating", "YouTube Marketing", "Amazon Partners", "Tik Tok Ads"]),
                coin=random.choice(CRYPTO_COINS)
            )
            
            # Add random greeting and signoff to make it look like a transcript
            greeting = random.choice(GREETINGS) if random.random() > 0.3 else ""
            signoff = random.choice(SIGNOFFS) if random.random() > 0.3 else ""
            
            parts = [p for p in [greeting, formatted_text, signoff] if p]
            full_transcript = " ".join(parts)
            
            # Let's clean up punctuation spaces
            full_transcript = full_transcript.replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")
            
            dataset.append({"text": full_transcript, "label": label})
            
    # Shuffle dataset to mix up classes
    random.shuffle(dataset)
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(dataset)
        
    print(f"Generated {len(dataset)} synthetic transcripts at '{output_path}'")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    output_file = project_root / "data" / "raw" / "synthetic_scam_data.csv"
    generate_dataset(output_file, samples_per_class=60)
