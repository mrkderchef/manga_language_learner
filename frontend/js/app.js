/**
 * App - SPA Router und Initialisierung
 */
(() => {
    const pages = document.querySelectorAll('.page');
    const navLinks = document.querySelectorAll('.nav-link');
    const modeCards = document.querySelectorAll('.mode-card');

    function navigate(pageName) {
        pages.forEach(p => p.classList.remove('active'));
        navLinks.forEach(l => l.classList.remove('active'));

        const target = document.getElementById(`page-${pageName}`);
        if (target) target.classList.add('active');

        const activeLink = document.querySelector(`[data-page="${pageName}"]`);
        if (activeLink) activeLink.classList.add('active');

        if (pageName === 'scanner') Scanner.loadPanels();
        if (pageName === 'learning') Learning.loadPanels();
    }

    // Nav link clicks
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            navigate(link.dataset.page);
        });
    });

    // Mode card clicks on home page
    modeCards.forEach(card => {
        card.addEventListener('click', () => {
            navigate(card.dataset.goto);
        });
    });

    Scanner.init();
    Learning.init();
})();
