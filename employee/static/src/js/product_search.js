odoo.define('website_sale_autocomplete_search.product_search', function (require) {
    "use strict";
    
    $(function() {
        $("#oe_search_box_data").autocomplete({
            source: function(request, response) {
                $.ajax({
                url: "/employee/skill",
                method: "GET",
                dataType: "json",
                data: { keywords: request.term},
                success: function( data ) {
                    response( $.map( data, function( item ) {
                        return {
                            value: item.name,
                            id: item.id,
                        }
                    }));
                },
                error: function (error) {
                    console.error(error);               
                }
                });
            },
             select:function(suggestion, term, item){
                 $(this).prev('#skill_id').val(term.item.id);
             },
             minLength: 1
        }),
        $("#oe_search_school").autocomplete({
            source: function(request, response) {
                $.ajax({
                url: "/employee/school",
                method: "GET",
                dataType: "json",
                data: { keywords: request.term},
                success: function( data ) {
                    response( $.map( data, function( item ) {
                        return {
                            value: item.name,
                            id: item.id,
                        }
                    }));
                },
                error: function (error) {
                    console.error(error);               
                }
                });
            },
             select:function(suggestion, term, item){
                 $(this).prev('#school_id').val(term.item.id);
             },
             minLength: 1
        }),
        $("#oe_search_country").autocomplete({
            source: function(request, response) {
                $.ajax({
                url: "/employee/country",
                method: "GET",
                dataType: "json",
                data: { keywords: request.term},
                success: function( data ) {
                    response( $.map( data, function( item ) {
                        return {
                            value: item.name,
                            id: item.id,
                        }
                    }));
                },
                error: function (error) {
                    console.error(error);               
                }
                });
            },
             select:function(suggestion, term, item){
                 $(this).prev('#country_id').val(term.item.id);
             },
             minLength: 1
        }),
        $("#oe_search_state").autocomplete({
            source: function(request, response) {
                $.ajax({
                url: "/employee/state",
                method: "GET",
                dataType: "json",
                data: { keywords: request.term},
                success: function( data ) {
                    response( $.map( data, function( item ) {
                        return {
                            value: item.name,
                            id: item.id,
                        }
                    }));
                },
                error: function (error) {
                    console.error(error);               
                }
                });
            },
             select:function(suggestion, term, item){
                 $(this).prev('#state_id').val(term.item.id);
             },
             minLength: 1
        }),
        $("#oe_search_district").autocomplete({
            source: function(request, response) {
                $.ajax({
                url: "/employee/district",
                method: "GET",
                dataType: "json",
                data: { keywords: request.term},
                success: function( data ) {
                    response( $.map( data, function( item ) {
                        return {
                            value: item.name,
                            id: item.id,
                        }
                    }));
                },
                error: function (error) {
                    console.error(error);               
                }
                });
            },
             select:function(suggestion, term, item){
                 $(this).prev('#current_district').val(term.item.id);
             },
             minLength: 1
        }),
        $("#oe_search_home_country").autocomplete({
            source: function(request, response) {
                $.ajax({
                url: "/employee/country",
                method: "GET",
                dataType: "json",
                data: { keywords: request.term},
                success: function( data ) {
                    response( $.map( data, function( item ) {
                        return {
                            value: item.name,
                            id: item.id,
                        }
                    }));
                },
                error: function (error) {
                    console.error(error);               
                }
                });
            },
             select:function(suggestion, term, item){
                 $(this).prev('#home_country').val(term.item.id);
             },
             minLength: 1
        }),
        $("#oe_search_home_state").autocomplete({
            source: function(request, response) {
                $.ajax({
                url: "/employee/state",
                method: "GET",
                dataType: "json",
                data: { keywords: request.term},
                success: function( data ) {
                    response( $.map( data, function( item ) {
                        return {
                            value: item.name,
                            id: item.id,
                        }
                    }));
                },
                error: function (error) {
                    console.error(error);               
                }
                });
            },
             select:function(suggestion, term, item){
                 $(this).prev('#home_state').val(term.item.id);
             },
             minLength: 1
        }),
        $("#oe_search_home_disctrict").autocomplete({
            source: function(request, response) {
                $.ajax({
                url: "/employee/district",
                method: "GET",
                dataType: "json",
                data: { keywords: request.term},
                success: function( data ) {
                    response( $.map( data, function( item ) {
                        return {
                            value: item.name,
                            id: item.id,
                        }
                    }));
                },
                error: function (error) {
                    console.error(error);               
                }
                });
            },
             select:function(suggestion, term, item){
                 $(this).prev('#home_disctrict').val(term.item.id);
             },
             minLength: 1
        });

        var today = new Date();
        var dd = today.getDate();
        var mm = today.getMonth()+1; //January is 0!
        var yyyy = today.getFullYear();
        if(dd<10){
                dd='0'+dd
            } 
            if(mm<10){
                mm='0'+mm
            } 

        today = yyyy+'-'+mm+'-'+dd;
        document.getElementById("birthday").setAttribute("max", today);
        document.getElementById("id_date").setAttribute("max", today);
        document.getElementById("passport_date").setAttribute("max", today);
    });
    
    });