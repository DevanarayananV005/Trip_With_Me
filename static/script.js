function toggleDropdown() {
    var dropdownMenu = document.getElementById("dropdownMenu");
    dropdownMenu.style.display = dropdownMenu.style.display === "block" ? "none" : "block";
}

window.onclick = function(event) {
    if (!event.target.matches('.user-image')) {
        var dropdowns = document.getElementsByClassName("dropdown-menu");
        for (var i = 0; i < dropdowns.length; i++) {
            var openDropdown = dropdowns[i];
            if (openDropdown.style.display === "block") {
                openDropdown.style.display = "none";
            }
        }
    }
}

function logout() {
    // Implement logout functionality
    alert("Logged out!");
}

function toggleTheme() {
    var body = document.body;
    var themeIcon = document.getElementById("theme-icon");
    body.classList.toggle("light-mode");
    if (body.classList.contains("light-mode")) {
        themeIcon.classList.remove("fa-moon");
        themeIcon.classList.add("fa-sun");
    } else {
        themeIcon.classList.remove("fa-sun");
        themeIcon.classList.add("fa-moon");
    }
}

function toggleSidebar() {
    var sidebar = document.getElementById("sidebar");
    var content = document.getElementById("content");
    sidebar.classList.toggle("minimized");
    content.classList.toggle("shifted");
}

function showSection(sectionId) {
    var sections = document.querySelectorAll(".section");
    sections.forEach(function(section) {
        section.classList.remove("active");
    });
    document.getElementById(sectionId).classList.add("active");
}
