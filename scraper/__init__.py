from dotenv import load_dotenv
import os

load_dotenv(".env", override=True)

os.environ["PATH"] = os.environ["GECKODRIVER_DIR"] + ":" + os.environ["PATH"]
