$(document).ready(function() {
        // maybe ios fix!
        $('#tabs .tab-link').css('cursor','pointer');
        $('#tabs .tab-link').click(function() {
                console.log(this);
                var tab_id = $(this).attr('data-tab');
                console.log(tab_id);
                $('#tabs .tab-link').removeClass('current');
                $('.tab-content').removeClass('current');

                $(this).addClass('current');
                $("#" + tab_id).addClass('current');
        });
});
