from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, EmailField, BooleanField, SubmitField
from wtforms.validators import DataRequired, Length, Email, EqualTo, ValidationError, Optional

class LoginForm(FlaskForm):
    username_or_email = StringField(
        'Username or Email', 
        validators=[DataRequired(), Length(min=3, max=80)],
        render_kw={'placeholder': 'Enter your username or email'}
    )
    password = PasswordField(
        'Password', 
        validators=[DataRequired()],
        render_kw={'placeholder': 'Enter your password'}
    )
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class SignupForm(FlaskForm):
    username = StringField(
        'Username', 
        validators=[DataRequired(), Length(min=3, max=20)],
        render_kw={'placeholder': 'Choose a unique username'}
    )
    email = EmailField(
        'Email Address', 
        validators=[DataRequired(), Email()],
        render_kw={'placeholder': 'Enter your email address'}
    )
    password = PasswordField(
        'Password', 
        validators=[DataRequired(), Length(min=6)],
        render_kw={'placeholder': 'Create a strong password'}
    )
    confirm_password = PasswordField(
        'Confirm Password', 
        validators=[DataRequired(), EqualTo('password', message='Passwords must match')],
        render_kw={'placeholder': 'Confirm your password'}
    )
    agree_terms = BooleanField(
        'I agree to the Terms of Service and Privacy Policy', 
        validators=[DataRequired()]
    )
    submit = SubmitField('Create Account')

class ProfileForm(FlaskForm):
    username = StringField(
        'Username', 
        validators=[DataRequired(), Length(min=3, max=20)],
        render_kw={'placeholder': 'Your username'}
    )
    email = EmailField(
        'Email Address', 
        validators=[DataRequired(), Email()],
        render_kw={'placeholder': 'Your email address'}
    )
    current_password = PasswordField(
        'Current Password', 
        validators=[Optional()],
        render_kw={'placeholder': 'Enter current password to change password'}
    )
    new_password = PasswordField(
        'New Password', 
        validators=[Optional(), Length(min=6)],
        render_kw={'placeholder': 'Enter new password (optional)'}
    )
    confirm_new_password = PasswordField(
        'Confirm New Password', 
        validators=[Optional(), EqualTo('new_password', message='New passwords must match')],
        render_kw={'placeholder': 'Confirm new password'}
    )
    submit = SubmitField('Update Profile')

    def validate_new_password(self, new_password):
        if new_password.data and not self.current_password.data:
            raise ValidationError('Current password is required to set a new password.')

class AddUserForm(FlaskForm):
    username = StringField(
        'Username', 
        validators=[DataRequired(), Length(min=3, max=20)],
        render_kw={'placeholder': 'Enter username'}
    )
    email = EmailField(
        'Email Address', 
        validators=[DataRequired(), Email()],
        render_kw={'placeholder': 'Enter email address'}
    )
    password = PasswordField(
        'Password', 
        validators=[DataRequired(), Length(min=6)],
        render_kw={'placeholder': 'Enter password'}
    )
    submit = SubmitField('Add User')

class EditUserForm(FlaskForm):
    username = StringField(
        'Username', 
        validators=[DataRequired(), Length(min=3, max=20)],
        render_kw={'placeholder': 'Enter username'}
    )
    email = EmailField(
        'Email Address', 
        validators=[DataRequired(), Email()],
        render_kw={'placeholder': 'Enter email address'}
    )
    password = PasswordField(
        'New Password', 
        validators=[Optional(), Length(min=6)],
        render_kw={'placeholder': 'Leave empty to keep current password'}
    )
    submit = SubmitField('Update User')