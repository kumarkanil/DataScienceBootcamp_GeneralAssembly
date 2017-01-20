// Javascript for SEP

$(document).ready(function() {
	// if Javascript is enabled, do these things:
	$('body').removeClass('nojs');
	$('.nav-collapse .dropdown').removeClass('open');
	$('.nav-collapse').removeClass('in');
	$('#article-sidebar .btn-navbar').addClass('collapsed');
	$('#article-sidebar #article-nav').removeClass('in');
	$('#mirrors .btn-group').removeClass('open');
	
	// Collapse sidebar menu while jumping:
	$('#article-nav a').click(function(){
		$('#article-nav').removeClass('in');
		$('#article-nav').css('height','0');
	});
});
