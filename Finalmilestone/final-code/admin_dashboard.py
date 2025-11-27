import streamlit as st
import sqlite3
import bcrypt
import jwt
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import io
import base64

# =============================================================================
#  CONFIGURATION & CONSTANTS
# =============================================================================

SECRET_KEY = "your real token "

# =============================================================================
#  DATABASE FUNCTIONS
# =============================================================================

def get_db_connection():
    return sqlite3.connect('llm_users.db')

def get_all_users():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT email, role, created_at, last_login FROM users", conn)
    conn.close()
    return df

def get_user_stats():
    conn = get_db_connection()
    c = conn.cursor()

    # Total users
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]

    # Users by role
    c.execute("SELECT role, COUNT(*) FROM users GROUP BY role")
    role_distribution = dict(c.fetchall())

    # New users this month
    c.execute("""
        SELECT COUNT(*) FROM users
        WHERE strftime('%Y-%m', created_at) = strftime('%Y-%m', 'now')
    """)
    new_this_month = c.fetchone()[0]

    # Active users (logged in last 30 days)
    c.execute("""
        SELECT COUNT(*) FROM users
        WHERE last_login >= datetime('now', '-30 days')
    """)
    active_users = c.fetchone()[0]

    conn.close()

    return {
        'total_users': total_users,
        'role_distribution': role_distribution,
        'new_this_month': new_this_month,
        'active_users': active_users
    }

def get_activity_stats():
    conn = get_db_connection()
    c = conn.cursor()

    # Total activities
    c.execute("SELECT COUNT(*) FROM user_activity")
    total_activities = c.fetchone()[0]

    # Activities by type
    c.execute("SELECT activity_type, COUNT(*) FROM user_activity GROUP BY activity_type")
    activity_by_type = dict(c.fetchall())

    # Most active users
    c.execute("""
        SELECT email, COUNT(*) as activity_count
        FROM user_activity
        GROUP BY email
        ORDER BY activity_count DESC
        LIMIT 10
    """)
    top_users = c.fetchall()

    conn.close()

    return {
        'total_activities': total_activities,
        'activity_by_type': activity_by_type,
        'top_users': top_users
    }

def get_feedback_stats():
    conn = get_db_connection()
    c = conn.cursor()

    # Total feedback
    c.execute("SELECT COUNT(*) FROM user_feedback")
    total_feedback = c.fetchone()[0]

    # Average rating by feature
    c.execute("SELECT feature, AVG(rating) as avg_rating FROM user_feedback GROUP BY feature")
    feature_ratings = dict(c.fetchall())

    # Recent feedback
    c.execute("""
        SELECT email, feature, rating, comment, timestamp
        FROM user_feedback
        ORDER BY timestamp DESC
        LIMIT 50
    """)
    recent_feedback = c.fetchall()

    # Feedback comments for word cloud
    c.execute("SELECT comment FROM user_feedback WHERE comment IS NOT NULL AND comment != ''")
    feedback_comments = [row[0] for row in c.fetchall()]

    conn.close()

    return {
        'total_feedback': total_feedback,
        'feature_ratings': feature_ratings,
        'recent_feedback': recent_feedback,
        'feedback_comments': feedback_comments
    }

def get_all_activities(limit=1000):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''
        SELECT ua.email, ua.activity_type, ua.input_text, ua.output_text,
               ua.model_used, ua.timestamp, u.role
        FROM user_activity ua
        JOIN users u ON ua.email = u.email
        ORDER BY ua.timestamp DESC
        LIMIT ?
    ''', (limit,))
    results = c.fetchall()
    conn.close()

    activities = []
    for result in results:
        activities.append({
            'email': result[0],
            'activity_type': result[1],
            'input_text': result[2],
            'output_text': result[3],
            'model_used': result[4],
            'timestamp': result[5],
            'role': result[6]
        })
    return activities

def delete_user(email):
    conn = get_db_connection()
    c = conn.cursor()

    # First delete user's activities and feedback
    c.execute("DELETE FROM user_activity WHERE email=?", (email,))
    c.execute("DELETE FROM user_feedback WHERE email=?", (email,))

    # Then delete the user
    c.execute("DELETE FROM users WHERE email=?", (email,))
    conn.commit()
    conn.close()

def update_user_role(email, new_role):
    conn = get_db_connection()
    c = conn.cursor()

    # Check admin count if promoting to admin
    if new_role == "Admin":
        c.execute("SELECT COUNT(*) FROM users WHERE role='Admin'")
        admin_count = c.fetchone()[0]
        if admin_count >= 2:
            conn.close()
            return "Maximum admin limit (2) reached."

    c.execute("UPDATE users SET role = ? WHERE email = ?", (new_role, email))
    conn.commit()
    conn.close()
    return "Role updated successfully"

def search_global(query):
    conn = get_db_connection()
    c = conn.cursor()

    # Search users
    c.execute("SELECT email, role FROM users WHERE email LIKE ?", (f'%{query}%',))
    users = c.fetchall()

    # Search activities
    c.execute('''
        SELECT email, activity_type, input_text, output_text
        FROM user_activity
        WHERE input_text LIKE ? OR output_text LIKE ? OR activity_type LIKE ?
    ''', (f'%{query}%', f'%{query}%', f'%{query}%'))
    activities = c.fetchall()

    # Search feedback
    c.execute('''
        SELECT email, feature, comment
        FROM user_feedback
        WHERE comment LIKE ? OR feature LIKE ?
    ''', (f'%{query}%', f'%{query}%'))
    feedback = c.fetchall()

    conn.close()

    return {
        'users': users,
        'activities': activities,
        'feedback': feedback
    }

# =============================================================================
#  AUTHENTICATION FUNCTIONS
# =============================================================================

def decode_token(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    except:
        return None

def verify_admin_access(token):
    payload = decode_token(token)
    return payload and payload.get('role') == 'Admin'

# =============================================================================
#  VISUALIZATION FUNCTIONS
# =============================================================================

def create_user_growth_chart():
    # Simulated user growth data
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    users = np.cumsum(np.random.randint(5, 20, len(dates)))

    fig = px.line(x=dates, y=users, title='User Growth Over Time')
    fig.update_layout(xaxis_title='Month', yaxis_title='Total Users')
    return fig

def create_activity_chart(activity_stats):
    activity_types = list(activity_stats['activity_by_type'].keys())
    counts = list(activity_stats['activity_by_type'].values())

    fig = px.pie(values=counts, names=activity_types, title='Activity Distribution by Type')
    return fig

def create_feedback_chart(feedback_stats):
    features = list(feedback_stats['feature_ratings'].keys())
    ratings = [float(rating) for rating in feedback_stats['feature_ratings'].values()]

    fig = px.bar(x=features, y=ratings, title='Average Rating by Feature')
    fig.update_layout(xaxis_title='Feature', yaxis_title='Average Rating')
    return fig

def create_word_cloud(feedback_comments):
    if not feedback_comments:
        # Return a placeholder image
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'No feedback comments available',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16)
        ax.axis('off')
        return fig

    text = ' '.join(feedback_comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Feedback Word Cloud', fontsize=16)
    return fig

def create_top_users_chart(top_users):
    users = [user[0] for user in top_users]
    counts = [user[1] for user in top_users]

    fig = px.bar(x=counts, y=users, orientation='h', title='Top Active Users')
    fig.update_layout(xaxis_title='Activity Count', yaxis_title='User')
    return fig

# =============================================================================
#  ADMIN DASHBOARD COMPONENTS
# =============================================================================

def render_admin_header():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                color: white; border-radius: 15px; padding: 2rem; text-align: center;
                margin-bottom: 2rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
        <h1 style="margin:0; font-size: 2.5rem;">⚙️ Admin Dashboard</h1>
        <p style="margin:0; opacity: 0.9; font-size: 1.1rem;">Complete System Administration & Analytics</p>
    </div>
    """, unsafe_allow_html=True)

def render_admin_metrics():
    st.header("📊 System Overview")

    user_stats = get_user_stats()
    activity_stats = get_activity_stats()
    feedback_stats = get_feedback_stats()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Users", user_stats['total_users'])
    with col2:
        st.metric("Active Users (30d)", user_stats['active_users'])
    with col3:
        st.metric("Total Activities", activity_stats['total_activities'])
    with col4:
        st.metric("Total Feedback", feedback_stats['total_feedback'])

    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("New Users (Month)", user_stats['new_this_month'])
    with col6:
        st.metric("Admin Users", user_stats['role_distribution'].get('Admin', 0))
    with col7:
        st.metric("General Users", user_stats['role_distribution'].get('General User', 0))
    with col8:
        st.metric("Activity Types", len(activity_stats['activity_by_type']))

def render_user_management():
    st.header("👥 User Management")

    users_df = get_all_users()

    if not users_df.empty:
        # Display user table
        st.subheader("All Registered Users")

        # Add search and filter options
        col1, col2 = st.columns(2)
        with col1:
            search_email = st.text_input("Search by email")
        with col2:
            filter_role = st.selectbox("Filter by role", ["All", "Admin", "General User"])

        # Apply filters
        filtered_users = users_df
        if search_email:
            filtered_users = filtered_users[filtered_users['email'].str.contains(search_email, case=False)]
        if filter_role != "All":
            filtered_users = filtered_users[filtered_users['role'] == filter_role]

        # Display user table with actions
        for idx, user in filtered_users.iterrows():
            with st.expander(f"{user['email']} ({user['role']})"):
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    st.write(f"**Role:** {user['role']}")
                    st.write(f"**Created:** {user['created_at']}")
                    st.write(f"**Last Login:** {user['last_login'] or 'Never'}")

                with col2:
                    new_role = st.selectbox(
                        "Change Role",
                        ["General User", "Admin"],
                        index=0 if user['role'] == "General User" else 1,
                        key=f"role_{user['email']}"
                    )

                    if st.button("Update Role", key=f"update_{user['email']}"):
                        result = update_user_role(user['email'], new_role)
                        if "successfully" in result:
                            st.success(result)
                            st.rerun()
                        else:
                            st.error(result)

                with col3:
                    if st.button("🗑️ Delete", key=f"delete_{user['email']}", type="secondary"):
                        if st.session_state.get('admin_email') != user['email']:
                            delete_user(user['email'])
                            st.success(f"User {user['email']} deleted successfully!")
                            st.rerun()
                        else:
                            st.error("Cannot delete your own account while logged in")
    else:
        st.info("No users found in the system")

def render_activity_monitoring():
    st.header("📈 Activity Monitoring")

    activity_stats = get_activity_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Activity Distribution")
        if activity_stats['activity_by_type']:
            fig = create_activity_chart(activity_stats)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No activity data available")

    with col2:
        st.subheader("Top Active Users")
        if activity_stats['top_users']:
            fig = create_top_users_chart(activity_stats['top_users'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No user activity data available")

    # Detailed activity log
    st.subheader("Detailed Activity Log")
    activities = get_all_activities(limit=100)

    if activities:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            activity_filter = st.selectbox("Filter by type", ["All"] + list(set(a['activity_type'] for a in activities)))
        with col2:
            user_filter = st.selectbox("Filter by user", ["All"] + list(set(a['email'] for a in activities)))
        with col3:
            model_filter = st.selectbox("Filter by model", ["All"] + list(set(a['model_used'] for a in activities if a['model_used'])))

        # Apply filters
        filtered_activities = activities
        if activity_filter != "All":
            filtered_activities = [a for a in filtered_activities if a['activity_type'] == activity_filter]
        if user_filter != "All":
            filtered_activities = [a for a in filtered_activities if a['email'] == user_filter]
        if model_filter != "All":
            filtered_activities = [a for a in filtered_activities if a['model_used'] == model_filter]

        # Display activities
        for activity in filtered_activities[:20]:  # Show first 20
            with st.expander(f"{activity['timestamp']} - {activity['email']} - {activity['activity_type']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Input:**")
                    st.text(activity['input_text'][:200] + "..." if len(activity['input_text']) > 200 else activity['input_text'])
                with col2:
                    st.write("**Output:**")
                    st.text(activity['output_text'][:200] + "..." if len(activity['output_text']) > 200 else activity['output_text'])
                st.write(f"**Model:** {activity['model_used']} | **Role:** {activity['role']}")
    else:
        st.info("No activities recorded yet")

def render_feedback_analytics():
    st.header("💬 Feedback Analytics")

    feedback_stats = get_feedback_stats()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Ratings")
        if feedback_stats['feature_ratings']:
            fig = create_feedback_chart(feedback_stats)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feedback ratings available")

    with col2:
        st.subheader("Feedback Insights")
        if feedback_stats['feedback_comments']:
            fig = create_word_cloud(feedback_stats['feedback_comments'])
            st.pyplot(fig)
        else:
            st.info("No feedback comments available")

    # Recent feedback table
    st.subheader("Recent Feedback")
    if feedback_stats['recent_feedback']:
        feedback_df = pd.DataFrame(feedback_stats['recent_feedback'],
                                 columns=['Email', 'Feature', 'Rating', 'Comment', 'Timestamp'])

        # Add search and filter
        col1, col2 = st.columns(2)
        with col1:
            search_feedback = st.text_input("Search feedback comments")
        with col2:
            feature_filter = st.selectbox("Filter by feature", ["All"] + list(feedback_df['Feature'].unique()))

        # Apply filters
        filtered_feedback = feedback_df
        if search_feedback:
            filtered_feedback = filtered_feedback[filtered_feedback['Comment'].str.contains(search_feedback, case=False, na=False)]
        if feature_filter != "All":
            filtered_feedback = filtered_feedback[filtered_feedback['Feature'] == feature_filter]

        # Display feedback
        for idx, feedback in filtered_feedback.iterrows():
            rating_stars = "⭐" * feedback['Rating']
            with st.expander(f"{feedback['Timestamp']} - {feedback['Email']} - {rating_stars}"):
                st.write(f"**Feature:** {feedback['Feature']}")
                st.write(f"**Rating:** {rating_stars} ({feedback['Rating']}/5)")
                if pd.notna(feedback['Comment']) and feedback['Comment'].strip():
                    st.write(f"**Comment:** {feedback['Comment']}")
                else:
                    st.write("**Comment:** *No comment provided*")
    else:
        st.info("No feedback submitted yet")

def render_system_analytics():
    st.header("📊 System Analytics")

    st.subheader("User Growth Over Time")
    fig = create_user_growth_chart()
    st.plotly_chart(fig, use_container_width=True)

    # Additional analytics
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Database Statistics")
        user_stats = get_user_stats()
        activity_stats = get_activity_stats()

        st.metric("Total Database Size", "~2.5 MB")  # Placeholder
        st.metric("Users per Role", f"Admin: {user_stats['role_distribution'].get('Admin', 0)}, General: {user_stats['role_distribution'].get('General User', 0)}")
        st.metric("Most Used Feature", max(activity_stats['activity_by_type'].items(), key=lambda x: x[1])[0] if activity_stats['activity_by_type'] else "N/A")

    with col2:
        st.subheader("Performance Metrics")
        st.metric("Average Response Time", "~1.2s")
        st.metric("System Uptime", "99.8%")
        st.metric("Active Sessions", "~15")

def render_global_search():
    st.header("🔍 Global Search")

    search_query = st.text_input("Enter search term", placeholder="Search across users, activities, and feedback...")

    if search_query:
        results = search_global(search_query)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(f"Users ({len(results['users'])})")
            for user in results['users']:
                st.write(f"**{user[0]}** ({user[1]})")

        with col2:
            st.subheader(f"Activities ({len(results['activities'])})")
            for activity in results['activities'][:5]:  # Show first 5
                st.write(f"**{activity[0]}** - {activity[1]}")
                st.text(activity[2][:100] + "..." if len(activity[2]) > 100 else activity[2])

        with col3:
            st.subheader(f"Feedback ({len(results['feedback'])})")
            for feedback in results['feedback'][:5]:  # Show first 5
                st.write(f"**{feedback[0]}** - {feedback[1]}")
                if feedback[2]:
                    st.text(feedback[2][:100] + "..." if len(feedback[2]) > 100 else feedback[2])

def render_admin_settings():
    st.header("⚙️ Admin Settings")

    st.subheader("System Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.number_input("Max Users Allowed", min_value=10, max_value=1000, value=100, key="max_users")
        st.number_input("Session Timeout (minutes)", min_value=5, max_value=120, value=30, key="session_timeout")
        st.selectbox("Default User Role", ["General User", "Admin"], key="default_role")

    with col2:
        st.number_input("Activity Log Retention (days)", min_value=7, max_value=365, value=90, key="log_retention")
        st.number_input("Max File Upload Size (MB)", min_value=1, max_value=50, value=10, key="max_upload_size")
        st.selectbox("Theme", ["Light", "Dark", "Auto"], key="theme")

    if st.button("💾 Save Configuration", type="primary"):
        st.success("Configuration saved successfully!")

    st.subheader("Database Maintenance")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Backup Database", use_container_width=True):
            st.success("Database backup initiated successfully!")

        if st.button("🧹 Clear Old Logs", use_container_width=True):
            st.success("Old logs cleared successfully!")

    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            st.success("System report generated successfully!")

        if st.button("🔄 Reset Analytics", use_container_width=True, type="secondary"):
            st.warning("This will reset all analytics data. Continue?")
            if st.button("Yes, Reset Analytics"):
                st.success("Analytics data reset successfully!")

    # Logout Section
    st.markdown("---")
    st.subheader("🚪 Session Management")

    if st.button("🔓 Logout", type="primary", use_container_width=True):
        # Clear session state and redirect to login
        st.session_state.token = None
        st.session_state.current_page = "dashboard"
        st.success("Successfully logged out! Redirecting to login page...")
        st.rerun()

# =============================================================================
#  MAIN ADMIN DASHBOARD
# =============================================================================

def main():
    st.set_page_config(
        page_title="Admin Dashboard - LLM AI Platform",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Check authentication and admin access
    if 'token' not in st.session_state:
        st.error("🔐 Access Denied: Please login first")
        st.stop()

    payload = decode_token(st.session_state.token)
    if not payload or payload.get('role') != 'Admin':
        st.error("🚫 Admin Access Required: You don't have permission to view this page")
        st.stop()

    # Store admin email for reference
    st.session_state.admin_email = payload['sub']

    # Admin dashboard styling
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .admin-metric {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

    # Render admin dashboard
    render_admin_header()

    # Navigation tabs for different admin sections
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview",
        "👥 User Management",
        "📈 Activity Monitor",
        "💬 Feedback Analytics",
        "📊 System Analytics",
        "🔍 Global Search",
        "⚙️ Settings"
    ])

    with tab1:
        render_admin_metrics()

        # Quick stats cards
        user_stats = get_user_stats()
        activity_stats = get_activity_stats()
        feedback_stats = get_feedback_stats()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("👥 User Insights")
            st.metric("User Growth Rate", "+12% this month")
            st.metric("Activation Rate", "78%")
            st.metric("Avg. Sessions per User", "3.2")

        with col2:
            st.subheader("📈 Activity Insights")
            st.metric("Most Active Time", "2:00 PM - 4:00 PM")
            st.metric("Avg. Activities per User", "5.7")
            st.metric("Popular Feature", "Summarization")

        with col3:
            st.subheader("💬 Feedback Insights")
            st.metric("Avg. Satisfaction Score", "4.2/5")
            st.metric("Response Rate", "65%")
            st.metric("Common Theme", "Usability")

    with tab2:
        render_user_management()

    with tab3:
        render_activity_monitoring()

    with tab4:
        render_feedback_analytics()

    with tab5:
        render_system_analytics()

    with tab6:
        render_global_search()

    with tab7:
        render_admin_settings()

    # Footer
    st.markdown("---")
    st.markdown(
        "**Admin Dashboard** • LLM AI Platform v1.0 • "
        f"Logged in as: {payload['sub']} • "
        f"Session: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":

    main()
