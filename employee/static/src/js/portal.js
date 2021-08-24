odoo.define('hr_employee.employee_website_form', function (require) {
'use strict';

    require('web.dom_ready');

//
//    if (!$('.o_portal').length) {
//        return $.Deferred().reject("DOM doesn't contain '.o_portal'");
//    }

    if ($('.o_portal_details').length) {
        // curren addess
        var state_options = $("select[name='state_id']:enabled option:not(:first)");
        $('.o_portal_details').on('change', "select[name='country_id']", function () {
            var select = $("select[name='state_id']");
            state_options.detach();
            var displayed_state = state_options.filter("[data-country_id="+($(this).val() || 0)+"]");
            var nb = displayed_state.appendTo(select).show().size();
            select.parent().toggle(nb>=1);
        });
        $('.o_portal_details').find("select[name='country_id']").change();

        var distrcit_options = $("select[name='current_district']:enabled option:not(:first)");
        $('.o_portal_details').on('change', "select[name='state_id']", function () {
            var select = $("select[name='current_district']");
            distrcit_options.detach();
            var displayed_district = distrcit_options.filter("[data-state_id="+($(this).val() || 0)+"]");
            var nb = displayed_district.appendTo(select).show().size();
            select.parent().toggle(nb>=1);
        });
        $('.o_portal_details').find("select[name='state_id']").change();

        // home address
        var home_state_options = $("select[name='home_state']:enabled option:not(:first)");
        $('.o_portal_details').on('change', "select[name='home_country']", function () {
            var select = $("select[name='home_state']");
            home_state_options.detach();
            var displayed_home_state = home_state_options.filter("[data-home_country="+($(this).val() || 0)+"]");
            var nb = displayed_home_state.appendTo(select).show().size();
            select.parent().toggle(nb>=1);
        });
        $('.o_portal_details').find("select[name='home_country']").change();

        var home_disctrict_option = $("select[name='home_disctrict']:enabled option:not(:first)");
        $('.o_portal_details').on('change', "select[name='home_state']", function () {
            var select = $("select[name='home_disctrict']");
            home_disctrict_option.detach();
            var displayed_home_district = home_disctrict_option.filter("[data-home_state="+($(this).val() || 0)+"]");
            var nb = displayed_home_district.appendTo(select).show().size();
            select.parent().toggle(nb>=1);
        });
        $('.o_portal_details').find("select[name='home_state']").change();
    }
});
