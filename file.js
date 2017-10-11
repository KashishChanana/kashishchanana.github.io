/**
 * Created by home on 3/8/17.
 */
$(document).ready(function(){
    $('.parallax').parallax();
    $('.target').pushpin({
        top: 0,
        bottom: 1000,
        offset: 0
    });
    $('.collapsible').collapsible();
});
$('.pushpin-demo-nav').each(function() {
    var $this = $(this);
    var $target = $('#' + $(this).attr('data-target'));
    $this.pushpin({
        top: $target.offset().top,
        bottom: $target.offset().top + $target.outerHeight() - $this.height()
    });
});