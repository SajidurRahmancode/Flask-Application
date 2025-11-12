// Custom JavaScript for Flask Authentication App

document.addEventListener('DOMContentLoaded', function() {
    // Auto-dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        if (!alert.querySelector('.btn-close')) {
            setTimeout(() => {
                alert.style.opacity = '0';
                setTimeout(() => alert.remove(), 300);
            }, 5000);
        }
    });

    // Form validation
    const forms = document.querySelectorAll('form[data-validate="true"]');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            console.log('Main.js: Form submission intercepted');
            console.log('Form valid:', form.checkValidity());
            
            // Only prevent submission if form is actually invalid
            if (!form.checkValidity()) {
                console.log('Main.js: Preventing submission due to validation errors');
                e.preventDefault();
                e.stopPropagation();
                form.classList.add('was-validated');
                return false;
            }
            
            console.log('Main.js: Allowing form submission');
            form.classList.add('was-validated');
        });
    });

    // Password strength indicator
    const passwordInput = document.querySelector('#password');
    const passwordStrength = document.querySelector('#passwordStrength');
    
    if (passwordInput && passwordStrength) {
        passwordInput.addEventListener('input', function() {
            const password = this.value;
            const strength = calculatePasswordStrength(password);
            updatePasswordStrength(strength, passwordStrength);
        });
    }

    // Confirm password validation
    const confirmPasswordInput = document.querySelector('#confirmPassword');
    const passwordConfirmInput = document.querySelector('#password');
    
    if (confirmPasswordInput && passwordConfirmInput) {
        confirmPasswordInput.addEventListener('input', function() {
            if (this.value !== passwordConfirmInput.value) {
                this.setCustomValidity('Passwords do not match');
            } else {
                this.setCustomValidity('');
            }
        });
        
        passwordConfirmInput.addEventListener('input', function() {
            if (confirmPasswordInput.value !== this.value) {
                confirmPasswordInput.setCustomValidity('Passwords do not match');
            } else {
                confirmPasswordInput.setCustomValidity('');
            }
        });
    }

    // Loading states for buttons
    const loadingButtons = document.querySelectorAll('[data-loading]');
    loadingButtons.forEach(button => {
        button.addEventListener('click', function() {
            if (this.closest('form').checkValidity()) {
                showLoadingState(this);
            }
        });
    });

    // Edit user inline functionality
    const editButtons = document.querySelectorAll('.btn-edit');
    editButtons.forEach(button => {
        button.addEventListener('click', function() {
            const userId = this.dataset.userId;
            toggleEditMode(userId);
        });
    });

    // Delete confirmation
    const deleteButtons = document.querySelectorAll('.btn-delete');
    deleteButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            const userName = this.dataset.userName;
            if (!confirm(`Are you sure you want to delete user "${userName}"? This action cannot be undone.`)) {
                e.preventDefault();
            }
        });
    });
});

function calculatePasswordStrength(password) {
    let score = 0;
    const checks = {
        length: password.length >= 8,
        lowercase: /[a-z]/.test(password),
        uppercase: /[A-Z]/.test(password),
        number: /\d/.test(password),
        special: /[!@#$%^&*(),.?":{}|<>]/.test(password)
    };

    score = Object.values(checks).filter(check => check).length;
    
    if (score < 3) return 'weak';
    if (score < 5) return 'medium';
    return 'strong';
}

function updatePasswordStrength(strength, element) {
    const classes = ['weak', 'medium', 'strong'];
    element.classList.remove(...classes);
    element.classList.add(strength);
    
    const messages = {
        weak: 'Weak password',
        medium: 'Medium strength',
        strong: 'Strong password'
    };
    
    element.textContent = messages[strength];
}

function showLoadingState(button) {
    const originalText = button.textContent;
    button.disabled = true;
    button.innerHTML = `<span class="loading-spinner me-2"></span>Loading...`;
    
    // Reset after form submission
    setTimeout(() => {
        button.disabled = false;
        button.textContent = originalText;
    }, 3000);
}

function toggleEditMode(userId) {
    const row = document.querySelector(`tr[data-user-id="${userId}"]`);
    const isEditing = row.classList.contains('editing');
    
    if (isEditing) {
        // Save changes
        saveUserChanges(userId, row);
    } else {
        // Enter edit mode
        enterEditMode(userId, row);
    }
}

function enterEditMode(userId, row) {
    row.classList.add('editing');
    
    // Replace text with input fields
    const usernameCell = row.querySelector('.username');
    const emailCell = row.querySelector('.email');
    const statusCell = row.querySelector('.status');
    
    const username = usernameCell.textContent.trim();
    const email = emailCell.textContent.trim();
    const isActive = statusCell.textContent.trim() === 'Active';
    
    usernameCell.innerHTML = `<input type="text" class="form-control form-control-sm" value="${username}" data-field="username">`;
    emailCell.innerHTML = `<input type="email" class="form-control form-control-sm" value="${email}" data-field="email">`;
    statusCell.innerHTML = `
        <select class="form-select form-select-sm" data-field="is_active">
            <option value="true" ${isActive ? 'selected' : ''}>Active</option>
            <option value="false" ${!isActive ? 'selected' : ''}>Inactive</option>
        </select>
    `;
    
    // Update button
    const editBtn = row.querySelector('.btn-edit');
    editBtn.textContent = 'Save';
    editBtn.classList.replace('btn-edit', 'btn-success');
}

function saveUserChanges(userId, row) {
    const formData = new FormData();
    const inputs = row.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        formData.append(input.dataset.field, input.value);
    });
    
    fetch(`/users/${userId}/edit`, {
        method: 'POST',
        body: formData,
        headers: {
            'X-Requested-With': 'XMLHttpRequest'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            location.reload(); // Reload to show updated data
        } else {
            alert('Error updating user: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while updating the user.');
    });
}