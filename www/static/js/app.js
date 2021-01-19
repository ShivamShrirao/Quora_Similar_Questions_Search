$(document).ready(function(){
			
	$('#searchfield').autocomplete({
		source: "fetch",
		minLength: 1,
		select: function(event, ui){
			$('#searchfield').val(ui.item.value);
			window.location = ui.item.url;
		}
	}).data('ui-autocomplete')._renderItem = function(ul, item){
		return $("<li class='ui-autocomplete-row'></li>")
				.data("item.autocomplete", item)
				// .append("<a href='" + item.url + "'>" + item.label + "</a>")
				.append(item.score)
				.appendTo(ul);
		};

});