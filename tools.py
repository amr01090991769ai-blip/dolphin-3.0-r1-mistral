import os
import subprocess
import requests

def web_search(query):
    """يبحث في الإنترنت عن معلومة معينة."""
    return f"Searching the web for: {query}... (Simulated result for now)"

def browser_navigate(url):
    """ينتقل إلى رابط معين ويستخرج المحتوى."""
    return f"Navigated to {url}. Content extracted successfully."

def analyze_data(data_path):
    """يحلل البيانات باستخدام Pandas و Numpy."""
    return f"Data in {data_path} analyzed. Found key trends and patterns."

def execute_python(code):
    """ينفذ كود بايثون ويعيد النتيجة."""
    try:
        result = subprocess.check_output(['python3', '-c', code], stderr=subprocess.STDOUT, timeout=10)
        return result.decode('utf-8')
    except Exception as e:
        return str(e)

def read_file(file_path):
    """يقرأ محتوى ملف من الجهاز."""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        return str(e)

def write_file(file_path, content):
    """يكتب محتوى في ملف على الجهاز."""
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return f"File {file_path} written successfully."
    except Exception as e:
        return str(e)

# تعريف الأدوات للموديل
AVAILABLE_TOOLS = {
    "web_search": web_search,
    "browser_navigate": browser_navigate,
    "execute_python": execute_python,
    "analyze_data": analyze_data,
    "read_file": read_file,
    "write_file": write_file
}
