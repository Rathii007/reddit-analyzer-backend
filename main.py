from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncpraw
import asyncprawcore
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from groq import Groq
from datetime import datetime
import logging
from collections import Counter
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from typing import List, Dict, Optional

# Download NLTK data
nltk.download("vader_lexicon")
nltk.download("stopwords")
nltk.download("punkt")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://reddit-analyzer-frontend.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Async PRAW
reddit = asyncpraw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT"),
)

# Initialize Groq API
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# SQLite Database Setup
Base = declarative_base()
engine = create_engine("sqlite:///reddit.db", connect_args={"timeout": 15})
Session = sessionmaker(bind=engine)

# Database Models
class Post(Base):
    __tablename__ = "posts"
    id = Column(String, primary_key=True)
    title = Column(String)
    subreddit = Column(String)
    created_utc = Column(Float)
    username = Column(String)
    upvotes = Column(Integer)
    downvotes = Column(Integer)

class User(Base):
    __tablename__ = "users"
    username = Column(String, primary_key=True)
    link_karma = Column(Integer)
    comment_karma = Column(Integer)
    created_utc = Column(Float)

class UserComment(Base):
    __tablename__ = "user_comments"
    id = Column(String, primary_key=True)
    body = Column(String)
    subreddit = Column(String)
    created_utc = Column(Float)
    username = Column(String)
    upvotes = Column(Integer)
    downvotes = Column(Integer)

class Roast(Base):
    __tablename__ = "roasts"
    username = Column(String, primary_key=True)
    roast_text = Column(Text)
    created_at = Column(Float)

class SubredditRoast(Base):
    __tablename__ = "subreddit_roasts"
    subreddit = Column(String, primary_key=True)
    roast_text = Column(Text)
    created_at = Column(Float)

# Recreate database to ensure schema includes upvotes/downvotes
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# Pydantic Models
class SubredditRequest(BaseModel):
    subreddit: str

class UserRequest(BaseModel):
    username: str
    full: bool = False

class SubredditComparisonRequest(BaseModel):
    subreddit1: str
    subreddit2: str

class ToxicityRequest(BaseModel):
    username: str

class ViralPostPredictionRequest(BaseModel):
    username: str
    topic: str

# Helper Functions
async def fetch_user_data(username: str) -> Dict:
    session = Session()
    try:
        with session.no_autoflush:
            redditor = await reddit.redditor(username)
            if redditor is None:
                raise HTTPException(status_code=404, detail="User not found")

            try:
                await redditor.load()
                created_utc = redditor.created_utc if hasattr(redditor, "created_utc") else datetime.utcnow().timestamp()
                link_karma = redditor.link_karma if hasattr(redditor, "link_karma") else 0
                comment_karma = redditor.comment_karma if hasattr(redditor, "comment_karma") else 0
            except asyncprawcore.exceptions.NotFound:
                raise HTTPException(status_code=404, detail=f"User {username} not found")

            user = User(
                username=username,
                link_karma=link_karma,
                comment_karma=comment_karma,
                created_utc=created_utc,
            )
            session.merge(user)

            # Fetch all posts (no limit)
            posts = []
            async for submission in redditor.submissions.new(limit=None):
                post = Post(
                    id=submission.id,
                    title=submission.title,
                    subreddit=submission.subreddit.display_name,
                    created_utc=submission.created_utc,
                    username=username,
                    upvotes=submission.ups,
                    downvotes=submission.downs,
                )
                session.merge(post)
                posts.append({
                    "id": post.id,
                    "title": post.title,
                    "subreddit": post.subreddit,
                    "created_utc": post.created_utc,
                    "upvotes": post.upvotes,
                    "downvotes": post.downvotes,
                })

            # Fetch all comments (no limit)
            comments = []
            async for comment in redditor.comments.new(limit=None):
                comment_data = UserComment(
                    id=comment.id,
                    body=comment.body,
                    subreddit=comment.subreddit.display_name,
                    created_utc=comment.created_utc,
                    username=username,
                    upvotes=comment.ups,
                    downvotes=comment.downs,
                )
                session.merge(comment_data)
                comments.append({
                    "id": comment_data.id,
                    "body": comment_data.body,
                    "subreddit": comment_data.subreddit,
                    "created_utc": comment_data.created_utc,
                    "upvotes": comment_data.upvotes,
                    "downvotes": comment_data.downvotes,
                })

            session.commit()
            subreddits = {}
            for post in posts:
                sub = post["subreddit"]
                subreddits[sub] = subreddits.get(sub, 0) + 1
            for comment in comments:
                sub = comment["subreddit"]
                subreddits[sub] = subreddits.get(sub, 0) + 1

            return {
                "username": username,
                "link_karma": link_karma,
                "comment_karma": comment_karma,
                "created_utc": created_utc,
                "account_age_days": (datetime.utcnow().timestamp() - created_utc) / (24 * 3600),
                "posts": posts,
                "comments": comments,
                "subreddits": subreddits,
            }
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error fetching user data for {username}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    finally:
        session.close()

async def summarize_large_data(username: str, posts: List[Dict], comments: List[Dict], subreddits: Dict[str, int]) -> Dict:
    try:
        all_text = " ".join([p["title"] for p in posts] + [c["body"] for c in comments])[:5000]  # Cap for API
        top_subreddits = sorted(subreddits.items(), key=lambda x: x[1], reverse=True)[:3]
        total_activities = len(posts) + len(comments)

        prompt = (
            f"Summarize the Reddit activity of user {username} with {total_activities} activities. "
            f"Text sample (up to 5000 chars): '{all_text}'. "
            f"Top subreddits: {', '.join([f'r/{sub} ({count} activities)' for sub, count in top_subreddits])}. "
            f"Provide a concise summary (3-4 sentences) covering their main interests, engagement patterns, and subreddit focus. "
            f"Highlight key topics or behaviors, avoiding generic terms like 'it' or 'https'. Ensure the tone is engaging and specific."
        )

        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        summary = response.choices[0].message.content.strip()
        if not summary or len(summary.split(". ")) < 3:
            logger.warning(f"Incomplete summary for {username}, falling back to default.")
            summary = f"{username} is active in {', '.join([f'r/{sub}' for sub, _ in top_subreddits])}, focusing on various topics with {total_activities} activities."
        return {"summary": summary, "total_activities": total_activities}
    except Exception as e:
        logger.error(f"Error summarizing data for {username}: {str(e)}")
        return {"summary": f"{username}'s activity summary unavailable due to processing issues.", "total_activities": len(posts) + len(comments)}

async def generate_user_roast(username: str, user_data: dict, full: bool = False) -> str:
    session = Session()
    try:
        subreddits = user_data["subreddits"]
        top_subreddit = max(subreddits.items(), key=lambda x: x[1])[0] if subreddits else "none"
        total_karma = user_data["link_karma"] + user_data["comment_karma"]
        oldest_activity = min(
            [p["created_utc"] for p in user_data["posts"]] + [c["created_utc"] for c in user_data["comments"]],
            default=datetime.utcnow().timestamp()
        )
        account_age_years = (datetime.utcnow().timestamp() - oldest_activity) / (365 * 24 * 3600)
        sample_post = user_data["posts"][0]["title"] if user_data["posts"] else "no posts"
        sample_comment = user_data["comments"][0]["body"][:50] if user_data["comments"] else "no comments"

        all_text = " ".join([p["title"] for p in user_data["posts"]] + [c["body"] for c in user_data["comments"]]).lower()
        stop_words = set(stopwords.words("english"))
        words = [w for w in re.findall(r"\b\w+\b", all_text) if w not in stop_words and len(w) > 3]
        most_common_word = Counter(words).most_common(1)[0][0] if words else "nothing"

        prompt = (
            f"Create a witty, brutal, yet humorous roast (4-6 sentences) of Reddit user {username}. "
            f"User stats: {account_age_years:.1f} years on Reddit, {total_karma} total karma, most active in r/{top_subreddit}, "
            f"frequently uses the word '{most_common_word}', sample post: '{sample_post}', sample comment: '{sample_comment}'. "
            f"Mock their Reddit habits, like their karma obsession, subreddit loyalty, or repetitive word use, in a playful way. "
            f"Keep it Reddit-focused, avoid personal attacks beyond their online behavior, and ensure the tone is sharp, fun, and engaging. "
            f"End with a humorous call-to-action to improve their Reddit game."
        )

        max_attempts = 3
        roast_text = None
        for attempt in range(max_attempts):
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.9,
                )
                roast_text = response.choices[0].message.content.strip()
                if roast_text and len(roast_text.split(". ")) >= 4:
                    break
                logger.warning(f"Incomplete roast for {username}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Error generating roast for {username}: {str(e)}")

        if not roast_text or len(roast_text.split(". ")) < 4:
            roast_text = (
                f"{username}, a {account_age_years:.1f}-year Reddit veteran, is stuck in r/{top_subreddit} like it’s their hometown. "
                f"With {total_karma} karma, they’re clearly fishing for upvotes, but their posts like '{sample_post}' are barely making waves. "
                f"Obsessed with the word '{most_common_word}', their content is as repetitive as a karma farmer’s repost spree. "
                f"Time to step out of your comfort zone, champ—try a new subreddit or two!"
            )

        roast = Roast(
            username=username,
            roast_text=roast_text,
            created_at=datetime.utcnow().timestamp(),
        )
        session.merge(roast)
        session.commit()
        return roast_text
    except Exception as e:
        session.rollback()
        logger.error(f"Error generating user roast for {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate roast")
    finally:
        session.close()

async def generate_subreddit_roast(subreddit: str, posts: list) -> str:
    session = Session()
    try:
        if not posts:
            raise HTTPException(status_code=404, detail="No posts found for subreddit")

        oldest_post_years = (datetime.utcnow().timestamp() - min(p["created_utc"] for p in posts)) / (365 * 24 * 3600)
        all_text = " ".join(p["title"].lower() for p in posts)
        stop_words = set(stopwords.words("english"))
        words = [w for w in re.findall(r"\b\w+\b", all_text) if w not in stop_words and len(w) > 3]
        top_words = Counter(words).most_common(3)
        common_topic = top_words[0][0] if top_words else "generic content"
        secondary_topic = top_words[1][0] if len(top_words) > 1 else "nothing else"
        sample_post = posts[0]["title"] if posts else "no sample post"

        subreddit_obj = await reddit.subreddit(subreddit)
        description = subreddit_obj.description[:200] if hasattr(subreddit_obj, "description") else "no description"
        rules_mention = "rules" in description.lower() or "ban" in description.lower()
        rule_snippet = "with strict rules" if rules_mention else "with no clear direction"

        prompt = (
            f"Create a witty, brutal, yet humorous roast (4-6 sentences) of r/{subreddit}. "
            f"Subreddit stats: {oldest_post_years:.1f} years old, top topic: '{common_topic}', secondary topic: '{secondary_topic}', "
            f"sample post: '{sample_post}', description hint: {rule_snippet}. "
            f"Mock their age (too young or too old), repetitive content, and community quirks (e.g., strict rules or lack of focus) in a playful way. "
            f"Use relatable Reddit metaphors (e.g., 'newbie trying to mod', 'echo chamber') and ensure the tone is sharp, fun, and engaging. "
            f"End with a humorous suggestion for improvement."
        )

        max_attempts = 3
        roast_text = None
        for attempt in range(max_attempts):
            try:
                response = groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.9,
                )
                roast_text = response.choices[0].message.content.strip()
                if roast_text and len(roast_text.split(". ")) >= 4:
                    break
                logger.warning(f"Incomplete subreddit roast for r/{subreddit}, attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Error generating subreddit roast for r/{subreddit}: {str(e)}")

        if not roast_text or len(roast_text.split(". ")) < 4:
            roast_text = (
                f"Welcome to r/{subreddit}, a {oldest_post_years:.1f}-year-old subreddit that’s still figuring out its vibe {rule_snippet}. "
                f"Obsessed with '{common_topic}', their posts like '{sample_post}' are as repetitive as a karma farmer’s repost spree. "
                f"Even '{secondary_topic}' can’t save this echo chamber from feeling like a ghost town. "
                f"Time to spice things up—maybe try some fresh topics, or at least a new banner!"
            )

        roast = SubredditRoast(
            subreddit=subreddit,
            roast_text=roast_text,
            created_at=datetime.utcnow().timestamp(),
        )
        session.merge(roast)
        session.commit()
        return roast_text
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error generating subreddit roast for r/{subreddit}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate roast")
    finally:
        session.close()

# API Endpoints
@app.get("/")
async def read_root():
    return {
        "message": "Welcome to Reddit Analyzer!",
        "description": "Analyze subreddits, users, and more with our powerful API.",
        "version": "1.0.0",
        "endpoints": [
            {"path": "/posts", "description": "Get a roast and insights for a subreddit"},
            {"path": "/user", "description": "Fetch user data and a roast"},
            {"path": "/insights", "description": "Get detailed user insights"},
            {"path": "/reddit-therapist", "description": "Receive Reddit-focused advice"},
        ]
    }

@app.post("/posts")
async def get_posts(request: SubredditRequest):
    session = Session()
    try:
        subreddit = await reddit.subreddit(request.subreddit)
        posts = []
        async for submission in subreddit.hot(limit=50):
            post = Post(
                id=submission.id,
                title=submission.title,
                subreddit=request.subreddit,
                created_utc=submission.created_utc,
                username=submission.author.name if submission.author else "deleted",
                upvotes=submission.ups,
                downvotes=submission.downs,
            )
            session.merge(post)
            posts.append({
                "id": post.id,
                "title": post.title,
                "subreddit": post.subreddit,
                "created_utc": post.created_utc,
                "posted_at": datetime.utcfromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "username": post.username,
                "upvotes": post.upvotes,
                "downvotes": post.downvotes,
                "net_score": post.upvotes - post.downvotes
            })
        session.commit()

        roast = await generate_subreddit_roast(request.subreddit, posts)
        top_posts = sorted(posts, key=lambda x: x["net_score"], reverse=True)[:3]
        avg_engagement = round(np.mean([p["net_score"] for p in posts]), 2) if posts else 0
        total_posts = len(posts)

        return {
            "subreddit": request.subreddit,
            "roast": roast,
            "insights": {
                "total_posts_analyzed": total_posts,
                "average_engagement_score": avg_engagement,
                "top_posts": [
                    {
                        "title": p["title"],
                        "username": p["username"],
                        "net_score": p["net_score"],
                        "posted_at": p["posted_at"]
                    } for p in top_posts
                ],
                "summary": (
                    f"r/{request.subreddit} has {total_posts} recent posts with an average engagement score of {avg_engagement}. "
                    f"Top posts include '{top_posts[0]['title']}' by u/{top_posts[0]['username']} with {top_posts[0]['net_score']} net upvotes."
                    if top_posts else "No standout posts to highlight."
                )
            },
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    except asyncprawcore.exceptions.NotFound:
        raise HTTPException(status_code=404, detail=f"Subreddit {request.subreddit} not found")
    except Exception as e:
        session.rollback()
        logger.error(f"Error fetching posts for {request.subreddit}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.post("/sentiment")
async def get_sentiment(request: SubredditRequest):
    session = Session()
    try:
        posts = session.query(Post).filter_by(subreddit=request.subreddit).limit(10).all()
        if not posts:
            raise HTTPException(status_code=404, detail="No posts found for subreddit")

        sia = SentimentIntensityAnalyzer()
        results = [
            {
                "id": post.id,
                "title": post.title,
                "sentiment": "positive" if sia.polarity_scores(post.title)["compound"] > 0 else "negative" if sia.polarity_scores(post.title)["compound"] < 0 else "neutral",
                "sentiment_score": sia.polarity_scores(post.title)["compound"]
            }
            for post in posts
        ]

        sentiment_counts = Counter(r["sentiment"] for r in results)
        total_posts = len(results)
        positive_percentage = (sentiment_counts["positive"] / total_posts * 100) if total_posts else 0
        negative_percentage = (sentiment_counts["negative"] / total_posts * 100) if total_posts else 0
        neutral_percentage = (sentiment_counts["neutral"] / total_posts * 100) if total_posts else 0

        top_positive = max([r for r in results if r["sentiment"] == "positive"], key=lambda x: x["sentiment_score"], default=None)
        top_negative = min([r for r in results if r["sentiment"] == "negative"], key=lambda x: x["sentiment_score"], default=None)

        return {
            "subreddit": request.subreddit,
            "sentiment_analysis": results,
            "summary": {
                "total_posts_analyzed": total_posts,
                "sentiment_distribution": {
                    "positive": round(positive_percentage, 2),
                    "negative": round(negative_percentage, 2),
                    "neutral": round(neutral_percentage, 2)
                },
                "highlight": (
                    f"r/{request.subreddit} leans {max(sentiment_counts, key=sentiment_counts.get)} with {round(max(positive_percentage, negative_percentage, neutral_percentage), 2)}% of posts. "
                    f"Most positive post: '{top_positive['title']}' (score: {top_positive['sentiment_score']})" if top_positive else ""
                    f"Most negative post: '{top_negative['title']}' (score: {top_negative['sentiment_score']})" if top_negative else ""
                ).strip()
            },
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    except Exception as e:
        session.rollback()
        logger.error(f"Error analyzing sentiment for {request.subreddit}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.post("/user")
async def get_user_data(request: UserRequest):
    user_data = await fetch_user_data(request.username)
    roast = await generate_user_roast(request.username, user_data, full=request.full)

    top_subreddit = max(user_data["subreddits"].items(), key=lambda x: x[1])[0] if user_data["subreddits"] else "none"
    total_activities = len(user_data["posts"]) + len(user_data["comments"])
    top_post = max(user_data["posts"], key=lambda x: x["upvotes"], default=None)
    top_comment = max(user_data["comments"], key=lambda x: x["upvotes"], default=None)

    return {
        "username": request.username,
        "user_summary": {
            "account_age_years": round(user_data["account_age_days"] / 365, 1),
            "total_karma": user_data["link_karma"] + user_data["comment_karma"],
            "top_subreddit": f"r/{top_subreddit}",
            "total_activities": total_activities
        },
        "roast": roast,
        "highlights": {
            "top_post": {
                "title": top_post["title"],
                "subreddit": f"r/{top_post['subreddit']}",
                "upvotes": top_post["upvotes"],
                "posted_at": datetime.utcfromtimestamp(top_post["created_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC")
            } if top_post else "No posts found",
            "top_comment": {
                "body": top_comment["body"][:100] + "..." if len(top_comment["body"]) > 100 else top_comment["body"],
                "subreddit": f"r/{top_comment['subreddit']}",
                "upvotes": top_comment["upvotes"],
                "posted_at": datetime.utcfromtimestamp(top_comment["created_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC")
            } if top_comment else "No comments found"
        },
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.post("/roast")
async def get_user_roast(request: UserRequest):
    user_data = await fetch_user_data(request.username)
    roast_text = await generate_user_roast(request.username, user_data, full=False)

    top_subreddit = max(user_data["subreddits"].items(), key=lambda x: x[1])[0] if user_data["subreddits"] else "none"
    total_karma = user_data["link_karma"] + user_data["comment_karma"]

    return {
        "username": request.username,
        "roast": roast_text,
        "context": {
            "based_on": {
                "account_age_years": round(user_data["account_age_days"] / 365, 1),
                "total_karma": total_karma,
                "top_subreddit": f"r/{top_subreddit}",
                "total_activities": len(user_data["posts"]) + len(user_data["comments"])
            },
            "note": "This roast is based on your Reddit activity, focusing on your posting habits and subreddit engagement."
        },
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.post("/insights")
async def get_user_insights(request: UserRequest):
    user_data = await fetch_user_data(request.username)
    
    sia = SentimentIntensityAnalyzer()
    
    total_activities = len(user_data["posts"]) + len(user_data["comments"])
    summary_data = None
    if total_activities > 500:
        summary_data = await summarize_large_data(
            request.username,
            user_data["posts"],
            user_data["comments"],
            user_data["subreddits"]
        )
    
    stop_words = set(stopwords.words("english") + ["https", "redd", "preview", "com", "www"])
    all_text = " ".join([p["title"] for p in user_data["posts"]] + [c["body"] for c in user_data["comments"]]).lower()
    words = [w for w in word_tokenize(all_text) if w.isalpha() and w not in stop_words and len(w) > 3]
    top_interests = Counter(words).most_common(5) if words else []
    
    controversial_comments = [
        {"id": c["id"], "body": c["body"], "subreddit": c["subreddit"], "downvotes": c["downvotes"]}
        for c in user_data["comments"] if c["downvotes"] > 10
    ]
    
    timestamps = [p["created_utc"] for p in user_data["posts"]] + [c["created_utc"] for c in user_data["comments"]]
    active_hours = Counter(datetime.utcfromtimestamp(ts).hour for ts in timestamps).most_common(3)
    active_days = Counter(datetime.utcfromtimestamp(ts).strftime("%A") for ts in timestamps).most_common(3)
    
    subreddit_engagement = {}
    for sub, count in user_data["subreddits"].items():
        sub_posts = [p for p in user_data["posts"] if p["subreddit"] == sub]
        sub_comments = [c for c in user_data["comments"] if c["subreddit"] == sub]
        total_upvotes = sum(p["upvotes"] for p in sub_posts) + sum(c["upvotes"] for c in sub_comments)
        total_downvotes = sum(p["downvotes"] for p in sub_posts) + sum(c["downvotes"] for c in sub_comments)
        subreddit_engagement[sub] = {
            "activity_count": count,
            "total_upvotes": total_upvotes,
            "total_downvotes": total_downvotes,
            "net_engagement": total_upvotes - total_downvotes
        }
    
    post_sentiments = [sia.polarity_scores(p["title"])["compound"] for p in user_data["posts"]]
    comment_sentiments = [sia.polarity_scores(c["body"])["compound"] for c in user_data["comments"]]
    avg_sentiment = np.mean(post_sentiments + comment_sentiments) if post_sentiments or comment_sentiments else 0
    sentiment_label = (
        "positive" if avg_sentiment > 0.1 else
        "negative" if avg_sentiment < -0.1 else
        "neutral"
    )
    
    account_age_years = user_data["account_age_days"] / 365
    total_posts = len(user_data["posts"])
    total_comments = len(user_data["comments"])
    total_karma = user_data["link_karma"] + user_data["comment_karma"]
    top_subreddit = max(user_data["subreddits"].items(), key=lambda x: x[1])[0] if user_data["subreddits"] else "none"

    top_post = max(user_data["posts"], key=lambda x: x["upvotes"], default=None)
    top_comment = max(user_data["comments"], key=lambda x: x["upvotes"], default=None)

    if not summary_data:
        summary = (
            f"{request.username} has been a Redditor for {account_age_years:.1f} years, amassing {total_karma} karma with {total_posts} posts and {total_comments} comments. "
            f"They're a regular in r/{top_subreddit}, where they’ve made a name for themselves with {user_data['subreddits'].get(top_subreddit, 0)} activities. "
            f"Their content has a {sentiment_label} vibe, with top interests including {', '.join(word for word, _ in top_interests[:3]) or 'various topics'}. "
            f"You’ll catch them posting most around {', '.join(f'{hour}:00 ({count} activities)' for hour, count in active_hours)} UTC, especially on {active_days[0][0] if active_days else 'various days'}."
        )
    else:
        summary = summary_data["summary"]

    subreddit_categories = {
        "IndianDankMemes": ["meme culture", "Indian pop culture", "humor"],
        "memes": ["meme culture", "humor"],
        "gaming": ["video games", "esports"],
        "technology": ["tech", "gadgets"],
        "movies": ["cinema", "pop culture"],
    }
    potential_interests = set()
    for sub in user_data["subreddits"]:
        potential_interests.update(subreddit_categories.get(sub, [sub.lower()]))

    return {
        "username": request.username,
        "insights": {
            "activity_summary": {
                "account_age_years": round(account_age_years, 1),
                "total_posts": total_posts,
                "total_comments": total_comments,
                "total_karma": total_karma,
                "data_summarized": summary_data is not None,
                "summary": summary,
                "total_activities": summary_data["total_activities"] if summary_data else total_activities,
            },
            "top_interests": [{"word": word, "count": count} for word, count in top_interests],
            "controversial_takes": controversial_comments or [{"message": "No controversial takes found. You're keeping it chill!"}],
            "posting_patterns": {
                "most_active_hours": [{"hour": hour, "count": count} for hour, count in active_hours],
                "most_active_days": [{"day": day, "count": count} for day, count in active_days],
            },
            "subreddit_engagement": [
                {"subreddit": sub, "stats": stats} for sub, stats in subreddit_engagement.items()
            ],
            "sentiment_analysis": {
                "average_sentiment": round(avg_sentiment, 2),
                "sentiment_label": sentiment_label,
                "description": (
                    f"Your posts and comments have a {sentiment_label} tone, with an average sentiment score of {round(avg_sentiment, 2)}. "
                    f"{'Keep spreading positivity!' if sentiment_label == 'positive' else 'Consider a more uplifting tone to boost engagement.' if sentiment_label == 'negative' else 'Your neutral tone is balanced—try adding some flair!'}"
                )
            },
            "potential_interests": list(potential_interests),
            "highlights": {
                "top_post": {
                    "title": top_post["title"],
                    "subreddit": f"r/{top_post['subreddit']}",
                    "upvotes": top_post["upvotes"],
                    "posted_at": datetime.utcfromtimestamp(top_post["created_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC")
                } if top_post else "No posts found",
                "top_comment": {
                    "body": top_comment["body"][:100] + "..." if len(top_comment["body"]) > 100 else top_comment["body"],
                    "subreddit": f"r/{top_comment['subreddit']}",
                    "upvotes": top_comment["upvotes"],
                    "posted_at": datetime.utcfromtimestamp(top_comment["created_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC")
                } if top_comment else "No comments found"
            }
        },
        "message": "Not enough content to analyze." if not (user_data["posts"] or user_data["comments"]) else None,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.post("/compare-subreddits")
async def compare_subreddits(request: SubredditComparisonRequest):
    session = Session()
    try:
        sub1_posts = session.query(Post).filter_by(subreddit=request.subreddit1).limit(100).all()
        sub2_posts = session.query(Post).filter_by(subreddit=request.subreddit2).limit(100).all()
        if not sub1_posts or not sub2_posts:
            raise HTTPException(status_code=404, detail="Subreddit posts not found")

        sia = SentimentIntensityAnalyzer()
        sub1_sentiment = np.mean([sia.polarity_scores(p.title)["compound"] for p in sub1_posts])
        sub2_sentiment = np.mean([sia.polarity_scores(p.title)["compound"] for p in sub2_posts])
        sub1_engagement = np.mean([p.upvotes for p in sub1_posts])
        sub2_engagement = np.mean([p.upvotes for p in sub2_posts])

        sentiment_winner = (
            request.subreddit1 if sub1_sentiment > sub2_sentiment else request.subreddit2
            if sub2_sentiment > sub1_sentiment else "neither (tie)"
        )
        engagement_winner = (
            request.subreddit1 if sub1_engagement > sub2_engagement else request.subreddit2
            if sub2_engagement > sub1_engagement else "neither (tie)"
        )

        return {
            "comparison": {
                "subreddit1": request.subreddit1,
                "subreddit2": request.subreddit2,
                "sentiment_comparison": {
                    request.subreddit1: round(sub1_sentiment, 2),
                    request.subreddit2: round(sub2_sentiment, 2)
                },
                "engagement_comparison": {
                    request.subreddit1: round(sub1_engagement, 2),
                    request.subreddit2: round(sub2_engagement, 2)
                }
            },
            "analysis": {
                "sentiment_winner": f"r/{sentiment_winner}",
                "engagement_winner": f"r/{engagement_winner}",
                "summary": (
                    f"r/{request.subreddit1} has a sentiment score of {round(sub1_sentiment, 2)} compared to r/{request.subreddit2}'s {round(sub2_sentiment, 2)}, "
                    f"making r/{sentiment_winner} the more positive community. "
                    f"Engagement-wise, r/{request.subreddit1} averages {round(sub1_engagement, 2)} upvotes per post, while r/{request.subreddit2} averages {round(sub2_engagement, 2)}, "
                    f"with r/{engagement_winner} taking the lead."
                )
            },
            "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        }
    except Exception as e:
        session.rollback()
        logger.error(f"Error comparing subreddits {request.subreddit1} and {request.subreddit2}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        session.close()

@app.post("/reddit-therapist")
async def reddit_therapist(request: UserRequest):
    user_data = await fetch_user_data(request.username)
    
    if not user_data.get("created_utc"):
        raise HTTPException(status_code=500, detail="Failed to retrieve user creation date")
    
    top_subreddit = max(user_data["subreddits"].items(), key=lambda x: x[1])[0] if user_data["subreddits"] else "none"
    total_karma = user_data["link_karma"] + user_data["comment_karma"]
    sample_post = user_data["posts"][0]["title"] if user_data["posts"] else "none"
    sample_comment = user_data["comments"][0]["body"][:100] if user_data["comments"] else "none"
    
    summary_data = None
    total_activities = len(user_data["posts"]) + len(user_data["comments"])
    if total_activities > 500:
        summary_data = await summarize_large_data(
            request.username,
            user_data["posts"],
            user_data["comments"],
            user_data["subreddits"]
        )
    
    summary_text = summary_data["summary"] if summary_data else (
        f"Active in r/{top_subreddit} with {total_activities} total activities."
    )
    
    sia = SentimentIntensityAnalyzer()
    comment_sentiments = [sia.polarity_scores(c["body"])["compound"] for c in user_data["comments"]]
    avg_sentiment = np.mean(comment_sentiments) if comment_sentiments else 0
    sentiment_label = "positive" if avg_sentiment > 0.1 else "negative" if avg_sentiment < -0.1 else "neutral"

    prompt = (
        f"Act as a friendly Reddit therapist for user {request.username}. Analyze their behavior and provide specific, constructive advice (6-8 sentences) to improve their Reddit experience. "
        f"User data: {total_karma} total karma, account age: {(datetime.utcnow().timestamp() - user_data['created_utc']) / (365 * 24 * 3600):.1f} years, "
        f"sentiment: {sentiment_label}, summary: '{summary_text}', sample post: '{sample_post}', sample comment: '{sample_comment}'. "
        f"Focus on their subreddit activity, posting/commenting style, emotional tone, and engagement patterns. Suggest ways to contribute more meaningfully, diversify their subreddit interactions, and enhance their content quality. "
        f"Ensure the advice is positive, actionable, and tailored to their Reddit behavior. Use a warm, empathetic tone and end with an encouraging note."
    )
    
    max_attempts = 5
    advice = None
    for attempt in range(max_attempts):
        try:
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            advice = response.choices[0].message.content.strip()
            
            if advice and advice[-1] in ".!?" and len(advice.split(". ")) >= 6:
                break
            logger.warning(f"Incomplete therapist response for {request.username}, attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Error generating therapist advice for {request.username}: {str(e)}")
        
        if attempt == max_attempts - 1:
            advice = (
                f"Hey {request.username}, your Reddit journey in r/{top_subreddit} shows dedication with {total_activities} activities. "
                f"Your posts like '{sample_post}' are a start, but they could use more flair to stand out. "
                f"Your comments, like '{sample_comment}', seem {sentiment_label}, so let’s channel that energy positively. "
                f"Try engaging in new subreddits to broaden your horizons and spark fresh ideas. "
                f"Share more unique content to boost your {total_karma} karma and connect with others. "
                f"Keep exploring, and you’ll find your Reddit groove—I believe in you!"
            )

    return {
        "username": request.username,
        "user_snapshot": {
            "total_karma": total_karma,
            "account_age_years": round((datetime.utcnow().timestamp() - user_data['created_utc']) / (365 * 24 * 3600), 1),
            "top_subreddit": f"r/{top_subreddit}",
            "sentiment": sentiment_label,
            "total_activities": total_activities
        },
        "advice": advice,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.post("/toxicity-score")
async def toxicity_score(request: ToxicityRequest):
    user_data = await fetch_user_data(request.username)
    comments = user_data["comments"]
    if not comments:
        raise HTTPException(status_code=404, detail="No comments found")

    swear_words = ["fuck", "shit", "asshole", "idiot", "moron"]
    toxic_comments = [c for c in comments if any(word in c["body"].lower() for word in swear_words)]
    toxicity_score = len(toxic_comments) / len(comments) * 100

    top_toxic_comments = sorted(toxic_comments, key=lambda x: x["downvotes"], reverse=True)[:3]

    return {
        "username": request.username,
        "toxicity_analysis": {
            "toxicity_score": round(toxicity_score, 2),
            "toxic_comments_count": len(toxic_comments),
            "total_comments": len(comments),
            "top_toxic_comments": [
                {
                    "body": c["body"][:100] + "..." if len(c["body"]) > 100 else c["body"],
                    "subreddit": f"r/{c['subreddit']}",
                    "downvotes": c["downvotes"],
                    "posted_at": datetime.utcfromtimestamp(c["created_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC")
                } for c in top_toxic_comments
            ],
            "summary": (
                f"u/{request.username} has a toxicity score of {round(toxicity_score, 2)}%, with {len(toxic_comments)} out of {len(comments)} comments flagged as toxic. "
                f"{'Consider adopting a more positive tone to improve your Reddit interactions.' if toxicity_score > 20 else 'You’re keeping it mostly positive—keep up the good vibes!'}"
            )
        },
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.post("/predict-viral-post")
async def predict_viral_post(request: ViralPostPredictionRequest):
    user_data = await fetch_user_data(request.username)
    top_subreddit = max(user_data["subreddits"].items(), key=lambda x: x[1])[0] if user_data["subreddits"] else "none"
    prompt = (
        f"Predict the likelihood of a Reddit post going viral for user {request.username} on topic '{request.topic}'. "
        f"User data: Past successful posts (over 100 upvotes): {[p['title'] for p in user_data['posts'] if p['upvotes'] > 100][:3] or ['none']}, "
        f"Most active subreddit: r/{top_subreddit}. "
        f"Estimate upvote potential (0-100%) and suggest 2-3 specific improvements (e.g., timing, title style, content type) in 4-5 sentences."
    )
    max_attempts = 3
    prediction = None
    for attempt in range(max_attempts):
        try:
            response = groq_client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            prediction = response.choices[0].message.content.strip()
            if prediction and len(prediction.split(". ")) >= 4:
                break
            logger.warning(f"Incomplete viral post prediction for {request.username}, attempt {attempt + 1}")
        except Exception as e:
            logger.error(f"Error predicting viral post for {request.username}: {str(e)}")

    if not prediction or len(prediction.split(". ")) < 4:
        prediction = (
            f"A post on '{request.topic}' by {request.username} in r/{top_subreddit} has a 50% chance of going viral. "
            f"Past successful posts like {[p['title'] for p in user_data['posts'] if p['upvotes'] > 100][:1] or ['none']} suggest some potential. "
            f"To improve, post during peak hours (e.g., 12:00 UTC) and use a catchy title with a question. "
            f"Add a meme or image to boost engagement."
        )

    successful_posts = [p for p in user_data["posts"] if p["upvotes"] > 100][:3]

    return {
        "username": request.username,
        "topic": request.topic,
        "prediction": {
            "text": prediction,
            "context": {
                "top_subreddit": f"r/{top_subreddit}",
                "past_successful_posts": [
                    {
                        "title": p["title"],
                        "upvotes": p["upvotes"],
                        "posted_at": datetime.utcfromtimestamp(p["created_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC")
                    } for p in successful_posts
                ] if successful_posts else "No past successful posts found.",
                "note": "Prediction based on your activity in your most active subreddit and past post performance."
            }
        },
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.post("/time-machine")
async def reddit_time_machine(request: UserRequest):
    user_data = await fetch_user_data(request.username)
    oldest_activity = min(
        [p["created_utc"] for p in user_data["posts"]] + [c["created_utc"] for c in user_data["comments"]],
        default=None
    )
    if not oldest_activity:
        raise HTTPException(status_code=404, detail="No activity found")

    oldest_post = min(user_data["posts"], key=lambda x: x["created_utc"], default={})
    oldest_comment = min(user_data["comments"], key=lambda x: x["created_utc"], default={})

    oldest_activity_date = datetime.utcfromtimestamp(oldest_activity).strftime("%Y-%m-%d")
    oldest_post_date = datetime.utcfromtimestamp(oldest_post["created_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC") if oldest_post else "N/A"
    oldest_comment_date = datetime.utcfromtimestamp(oldest_comment["created_utc"]).strftime("%Y-%m-%d %H:%M:%S UTC") if oldest_comment else "N/A"

    return {
        "username": request.username,
        "time_machine": {
            "oldest_activity_date": oldest_activity_date,
            "oldest_post": {
                "title": oldest_post.get("title", "No posts found"),
                "subreddit": f"r/{oldest_post['subreddit']}" if oldest_post else "N/A",
                "posted_at": oldest_post_date
            },
            "oldest_comment": {
                "body": oldest_comment.get("body", "No comments found")[:100] + "..." if oldest_comment and len(oldest_comment.get("body", "")) > 100 else oldest_comment.get("body", "No comments found"),
                "subreddit": f"r/{oldest_comment['subreddit']}" if oldest_comment else "N/A",
                "posted_at": oldest_comment_date
            },
            "narrative": (
                f"Let’s hop in the Reddit time machine, u/{request.username}! "
                f"Way back on {oldest_activity_date}, you kicked off your journey—your first post was '{oldest_post.get('title', 'nothing')}' in r/{oldest_post['subreddit'] if oldest_post else 'N/A'}. "
                f"Around the same time, you dropped your first comment in r/{oldest_comment['subreddit'] if oldest_comment else 'N/A'}, saying '{oldest_comment.get('body', 'nothing')[:50] + '...' if oldest_comment and len(oldest_comment.get('body', '')) > 50 else oldest_comment.get('body', 'nothing')}'. "
                f"What a throwback—look how far you’ve come since then!"
            )
        },
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

@app.post("/recommend-subreddits")
async def recommend_subreddits(request: UserRequest):
    user_data = await fetch_user_data(request.username)
    if not user_data["subreddits"]:
        raise HTTPException(status_code=404, detail="No subreddit activity found")

    stop_words = set(stopwords.words("english") + ["https", "redd", "preview", "com", "www"])
    all_text = " ".join([p["title"] for p in user_data["posts"]] + [c["body"] for c in user_data["comments"]]).lower()
    words = [w for w in word_tokenize(all_text) if w.isalpha() and w not in stop_words and len(w) > 3]
    top_interests = [word for word, _ in Counter(words).most_common(5)] if words else []

    subreddit_categories = {
        "IndianDankMemes": ["meme culture", "Indian pop culture", "humor"],
        "memes": ["meme culture", "humor"],
        "gaming": ["video games", "esports"],
        "technology": ["tech", "gadgets"],
        "movies": ["cinema", "pop culture"],
        "programming": ["coding", "software development"],
        "science": ["science", "research"],
        "photocritique": ["photography", "art"],
        "books": ["literature", "reading"],
        "fitness": ["health", "exercise"]
    }

    user_interests = set()
    for sub in user_data["subreddits"]:
        user_interests.update(subreddit_categories.get(sub, [sub.lower()]))

    popular_subs = ["programming", "technology", "gaming", "movies", "science", "photocritique", "books", "fitness"]
    user_subs = list(user_data["subreddits"].keys())
    recommendations = []
    for sub in popular_subs:
        if sub not in user_subs:
            sub_interests = subreddit_categories.get(sub, [sub.lower()])
            if any(interest in user_interests for interest in sub_interests) or any(interest in sub_interests for interest in top_interests):
                recommendations.append({
                    "subreddit": f"r/{sub}",
                    "reason": f"Matches your interest in {', '.join(set(sub_interests) & user_interests) or 'similar topics'}"
                })
        if len(recommendations) >= 3:
            break

    if not recommendations:
        recommendations = [
            {"subreddit": f"r/{sub}", "reason": "A popular subreddit to explore new topics"} for sub in popular_subs[:3] if sub not in user_subs
        ]

    return {
        "username": request.username,
        "recommendations": recommendations,
        "user_interests": list(user_interests),
        "summary": (
            f"Based on your activity, you’re into {', '.join(list(user_interests)[:3]) or 'various topics'}. "
            f"We recommend exploring {', '.join(rec['subreddit'] for rec in recommendations)} to diversify your Reddit experience!"
        ),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }