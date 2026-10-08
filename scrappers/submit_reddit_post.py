"""
submit_reddit_post.py — Secure Programmatic Reddit Posting Interface
─────────────────────────────────────────────────────────────────────────────
Uses official OAuth protocol wrappers to securely publish text entry segments.
"""
import praw

# Initialize the official secure OAuth handshake framework
reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID_HERE",          # Replace with your Client ID
    client_secret="YOUR_CLIENT_SECRET_HERE",  # Replace with your Client Secret
    user_agent="MindCare Academic Testing Script v1.0 by /u/YOUR_REDDIT_USERNAME",
    username="YOUR_REDDIT_USERNAME",          # Your exact Reddit login handle
    password="YOUR_REDDIT_PASSWORD"           # Your exact Reddit password
)

def submit_text_post(title: str, body: str):
    try:
        # Grabs your authorized username string
        me = reddit.user.me()
        print(f"🔒 Authenticated successfully as u/{me.name}")
        
        # Submitting directly to your own profile timeline page
        target_subreddit = f"u_{me.name}"
        
        print(f"📤 Submitting text segment to r/{target_subreddit}...")
        submission = reddit.subreddit(target_subreddit).submit(
            title=title,
            selftext=body
        )
        
        print(f"\n✅ Post successfully published!")
        print(f"🔗 Permanent URL Link: {submission.url}")
        
    except Exception as e:
        print(f"❌ Failed to submit post to the platform: {str(e)}")

if __name__ == "__main__":
    # Test text configuration
    test_title = "Reflections on Burnout and Academic Stress"
    test_body = (
        "Lately, I’ve just been feeling incredibly drained and tired all the time, "
        "no matter how much sleep I get. I find myself pulling away from my friends "
        "and losing interest in the hobbies I used to love. It feels like I'm just "
        "going through the motions every day without any real motivation."
    )
    
    submit_text_post(title=test_title, body=test_body)
