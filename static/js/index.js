// Index page

document.addEventListener('DOMContentLoaded', () => {
    const toggleNavButton = document.querySelector('.toggle-nav');
    const navLinks = document.querySelector('.nav-links');

    toggleNavButton.addEventListener('click', () => {
        navLinks.classList.toggle('active');
    });
});